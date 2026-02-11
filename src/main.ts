import { CreateWebWorkerMLCEngine, type MLCEngineInterface, type InitProgressReport } from "@mlc-ai/web-llm";
import "./style.css";

const DEBOUNCE_MS = 2500;
const MIN_TEXT_LENGTH = 20;
const MODEL_ID = "Llama-3.2-1B-Instruct-q4f32_1-MLC";
const STORAGE_KEY = "clear-editor-content";
const SAVE_DEBOUNCE_MS = 400;

type StoredEditor = { content: string; cursor: number };

const SYSTEM_PROMPT = `You are a thoughtful psychologist reviewing a person's reflective notes. Your role is to occasionally ask one insightful question that invites deeper self-reflection—not to advise or interpret, but to gently probe.

Given the following text, identify 1-2 sentences that seem especially meaningful or ripe for reflection. Then ask ONE short, open-ended question (1-2 sentences) that could help the writer think more deeply about what they've written.

Respond ONLY with valid JSON in this exact format, no other text:
{"question": "Your question here", "sentences": ["first sentence", "second sentence if applicable"]}`;

type Thought = { question: string; sentences: string[] };

/**
 * Parse LLM response that may be one JSON object, multiple JSON objects,
 * or mixed text. Extract only question and sentences; never return raw JSON for display.
 */
function parseThoughtResponse(raw: string): { question: string; sentences: string[] } {
  let question = "";
  const sentences: string[] = [];
  const str = raw.replace(/```json\s*|\s*```/g, "").trim();

  // Find all {...} substrings and try to parse each
  let depth = 0;
  let start = -1;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (str[i] === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        const chunk = str.slice(start, i + 1);
        try {
          const obj = JSON.parse(chunk) as { question?: string; sentences?: string[] };
          if (typeof obj.question === "string" && obj.question.trim()) {
            question = obj.question.trim();
          }
          if (Array.isArray(obj.sentences)) {
            for (const s of obj.sentences) {
              if (typeof s === "string" && s.trim()) sentences.push(s.trim());
            }
          }
        } catch {
          // ignore unparseable chunks
        }
        start = -1;
      }
    }
  }

  // If we found no JSON, try parsing the whole string once
  if (!question && sentences.length === 0) {
    try {
      const obj = JSON.parse(str) as { question?: string; sentences?: string[] };
      if (typeof obj.question === "string" && obj.question.trim()) {
        question = obj.question.trim();
      }
      if (Array.isArray(obj.sentences)) {
        for (const s of obj.sentences) {
          if (typeof s === "string" && s.trim()) sentences.push(s.trim());
        }
      }
    } catch {
      // If it looks like plain text (no braces), use as question only
      const trimmed = str.trim();
      if (trimmed && !trimmed.startsWith("{")) {
        question = trimmed;
      }
    }
  }

  return { question, sentences };
}

let thought: Thought | null = null;
let isGenerating = false;
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
let saveDebounceTimer: ReturnType<typeof setTimeout> | null = null;
let engine: MLCEngineInterface | null = null;
let engineReady = false;

const $ = (sel: string) => document.querySelector(sel) as HTMLElement | null;

const app = $("#app")!;

function setLoading(show: boolean, label = "Loading...") {
  const loadingEl = $("#loading");
  const loadingLabel = $("#loading-label");
  if (!loadingEl || !loadingLabel) return;
  loadingEl.hidden = !show;
  loadingLabel.textContent = label;
}

function setError(msg: string | null) {
  const errorEl = $("#error");
  if (!errorEl) return;
  errorEl.hidden = !msg;
  errorEl.textContent = msg || "";
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Normalize whitespace for matching: collapse runs to single space, trim */
function normalizeForMatch(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/**
 * Build normalized text and map from normalized index to original start index.
 * So origStart[i] = original index in text of the i-th character in normalized text.
 */
function buildNormalizedMap(text: string): { normalized: string; origStart: number[] } {
  const normalized: string[] = [];
  const origStart: number[] = [];
  let lastWasSpace = false;
  for (let i = 0; i < text.length; i++) {
    const isSpace = /\s/.test(text[i]);
    if (isSpace) {
      if (!lastWasSpace) {
        normalized.push(" ");
        origStart.push(i);
      }
      lastWasSpace = true;
    } else {
      normalized.push(text[i]);
      origStart.push(i);
      lastWasSpace = false;
    }
  }
  return { normalized: normalized.join(""), origStart };
}

/** Find first occurrence of sentence in text, with flexible whitespace. Returns [start, end] in original text or null. */
function findSentenceRange(text: string, sentence: string): [number, number] | null {
  const s = normalizeForMatch(sentence);
  if (!s) return null;
  const { normalized, origStart } = buildNormalizedMap(text);
  const idx = normalized.indexOf(s);
  if (idx < 0) return null;
  const endNorm = idx + s.length;
  if (endNorm > origStart.length) return null;
  const startOrig = origStart[idx];
  const endOrig = origStart[endNorm - 1] + 1;
  return [startOrig, endOrig];
}

function findRanges(text: string, sentences: string[]): [number, number][] {
  const ranges: [number, number][] = [];
  for (const raw of sentences) {
    const trimmed = raw.trim();
    if (!trimmed) continue;
    const range = findSentenceRange(text, trimmed);
    if (range) ranges.push(range);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [start, end] of ranges) {
    if (merged.length > 0 && start <= merged[merged.length - 1][1]) {
      const last = merged[merged.length - 1];
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

function applyHighlights(editor: HTMLElement, sentences: string[]) {
  const text = editor.innerText || "";
  if (!text.trim() || !sentences.length) return;
  const ranges = findRanges(text, sentences);
  if (ranges.length === 0) return;
  let html = "";
  let pos = 0;
  for (const [start, end] of ranges) {
    html += escapeHtml(text.slice(pos, start));
    html += '<span class="highlight">' + escapeHtml(text.slice(start, end)) + "</span>";
    pos = end;
  }
  html += escapeHtml(text.slice(pos));
  editor.innerHTML = html;
}

function clearHighlights(editor: HTMLElement) {
  const highlights = editor.querySelectorAll(".highlight");
  highlights.forEach((el) => {
    const text = document.createTextNode(el.textContent || "");
    el.parentNode?.replaceChild(text, el);
  });
}

function getEditorText(): string {
  const editor = $("#editor");
  return (editor?.innerText ?? "").trim();
}

/** Get current caret position as character offset from the start of the editor text. */
function getCursorOffset(editor: HTMLElement): number {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return 0;
  const range = sel.getRangeAt(0);
  if (!editor.contains(range.startContainer)) return 0;
  let offset = 0;
  const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
  let node: Node | null = walker.currentNode;
  while (node) {
    const len = (node.textContent ?? "").length;
    if (node === range.startContainer) {
      return offset + Math.min(range.startOffset, len);
    }
    offset += len;
    node = walker.nextNode();
  }
  return offset;
}

/** Set caret to the given character offset. */
function setCursorOffset(editor: HTMLElement, targetOffset: number): void {
  const sel = window.getSelection();
  if (!sel) return;
  let offset = 0;
  const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null);
  let node: Node | null = walker.currentNode;
  while (node) {
    const len = (node.textContent ?? "").length;
    if (offset + len >= targetOffset) {
      const range = document.createRange();
      range.setStart(node, Math.min(targetOffset - offset, len));
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
      return;
    }
    offset += len;
    node = walker.nextNode();
  }
  if (node === null && offset > 0) {
    const lastText = editor.lastChild?.nodeType === Node.TEXT_NODE ? editor.lastChild : null;
    if (lastText) {
      const range = document.createRange();
      range.setStart(lastText, (lastText.textContent ?? "").length);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }
}

function saveToStorage() {
  const editor = $("#editor");
  const text = editor?.innerText ?? "";
  try {
    if (text) {
      const cursor = editor ? getCursorOffset(editor) : 0;
      const stored: StoredEditor = { content: text, cursor: Math.min(cursor, text.length) };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // ignore quota or disabled storage
  }
}

function loadFromStorage(): StoredEditor | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === "object" && "content" in parsed && typeof (parsed as StoredEditor).content === "string") {
        const { content, cursor } = parsed as StoredEditor;
        const c = typeof cursor === "number" && cursor >= 0 ? Math.min(cursor, content.length) : 0;
        return { content, cursor: c };
      }
    } catch {
      // fall through to legacy string format
    }
    return { content: raw, cursor: 0 };
  } catch {
    return null;
  }
}

function scheduleSave() {
  if (saveDebounceTimer) clearTimeout(saveDebounceTimer);
  saveDebounceTimer = setTimeout(() => {
    saveDebounceTimer = null;
    saveToStorage();
  }, SAVE_DEBOUNCE_MS);
}

function renderThought() {
  const sidebar = $("#sidebar");
  const sidebarContent = $("#sidebar-content");
  const sidebarQuote = $("#sidebar-quote");
  const editor = $("#editor");
  if (!sidebarContent || !sidebarQuote || !sidebar) return;
  if (!thought) {
    sidebarContent.innerHTML = "";
    sidebarQuote.innerHTML = "";
    if (editor) clearHighlights(editor);
    return;
  }
  sidebarContent.innerHTML = "";
  const p = document.createElement("p");
  p.className = "thought-question";
  p.textContent = thought.question;
  sidebarContent.appendChild(p);
  sidebarQuote.innerHTML = "";
  if (thought.sentences && thought.sentences.length > 0) {
    thought.sentences.forEach((s) => {
      const block = document.createElement("blockquote");
      block.className = "thought-sentence";
      block.textContent = s;
      sidebarQuote.appendChild(block);
    });
    if (editor) applyHighlights(editor, thought.sentences);
  }
}

function clearThought() {
  thought = null;
  renderThought();
}

function triggerThought() {
  const t = getEditorText();
  if (t.length < MIN_TEXT_LENGTH) return;
  if (isGenerating || !engine) return;
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }

  isGenerating = true;
  setError(null);
  setLoading(true, "Thinking...");

  const messages = [
    { role: "system" as const, content: SYSTEM_PROMPT },
    { role: "user" as const, content: t },
  ];

  engine.chat.completions
    .create({ messages, temperature: 0.7, max_tokens: 256 })
    .then((reply) => {
      const raw = reply.choices[0]?.message?.content ?? "";
      const parsed = parseThoughtResponse(raw);
      thought = {
        question: parsed.question || "What stands out to you in what you wrote?",
        sentences: parsed.sentences,
      };
      renderThought();
    })
    .catch((err) => {
      setError(err?.message ?? "Something went wrong. Try again.");
    })
    .finally(() => {
      isGenerating = false;
      setLoading(false);
    });
}

function onTextInput() {
  scheduleSave();
  clearThought();
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    triggerThought();
  }, DEBOUNCE_MS);
}

function onGenAIClick() {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  triggerThought();
}

function initUI() {
  app.innerHTML = `
    <div class="layout">
      <div class="layout-spacer" aria-hidden="true"></div>
      <main class="main">
        <div
          id="editor"
          class="editor"
          contenteditable="true"
          data-placeholder="Write your thoughts here..."
          spellcheck="true"
          role="textbox"
          aria-multiline="true"
        ></div>
      </main>
      <aside id="sidebar" class="sidebar" aria-label="AI thought">
        <div id="sidebar-content" class="sidebar-content"></div>
        <div id="sidebar-quote" class="sidebar-quote"></div>
      </aside>
      <div id="loading" class="loading" hidden aria-live="polite">
        <span id="loading-label">Loading...</span>
      </div>
      <div id="error" class="error" hidden role="alert"></div>
      <button type="button" id="genai-btn" class="genai-btn" aria-label="Generate reflection">
        <svg class="genai-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"/>
        </svg>
      </button>
    </div>
  `;

  const newEditor = $("#editor");
  const newSidebar = $("#sidebar");
  const newSidebarContent = $("#sidebar-content");
  const newSidebarQuote = $("#sidebar-quote");
  const newLoading = $("#loading");
  const newLoadingLabel = $("#loading-label");
  const newGenaiBtn = $("#genai-btn");
  const newError = $("#error");

  if (!newEditor || !newSidebar || !newSidebarContent || !newSidebarQuote || !newLoading || !newLoadingLabel || !newGenaiBtn || !newError) return;

  const saved = loadFromStorage();
  if (saved?.content) {
    newEditor.textContent = saved.content;
    const cursor = saved.cursor;
    requestAnimationFrame(() => {
      setCursorOffset(newEditor, cursor);
    });
  }

  newEditor.addEventListener("input", onTextInput);
  newEditor.addEventListener("keydown", () => {
    if (debounceTimer) clearTimeout(debounceTimer);
  });
  newGenaiBtn.addEventListener("click", onGenAIClick);
}

async function initEngine() {
  initUI();

  if (!("gpu" in navigator)) {
    setError("Clear requires a browser with WebGPU support (e.g. Chrome).");
    return;
  }

  const initProgressCallback = (report: InitProgressReport) => {
    const loadingLabelEl = $("#loading-label");
    if (loadingLabelEl) loadingLabelEl.textContent = report.text;
  };

  try {
    setLoading(true, "Loading model...");
    engine = await CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      MODEL_ID,
      { initProgressCallback }
    );
    engineReady = true;
    const editorEl = $("#editor");
    if (editorEl) {
      editorEl.focus();
      const saved = loadFromStorage();
      if (saved?.content && typeof saved.cursor === "number") {
        requestAnimationFrame(() => setCursorOffset(editorEl, saved.cursor));
      }
    }
  } catch (err) {
    setError((err as Error)?.message ?? "Failed to load model. Try again.");
  } finally {
    setLoading(false);
  }

  const genaiBtnEl = $("#genai-btn") as HTMLButtonElement | null;
  if (genaiBtnEl) genaiBtnEl.disabled = !engineReady;
}

initEngine();
