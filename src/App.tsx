import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { BookMarked, Search, BookOpen, Settings, ArrowLeft, Key, ChevronDown, Copy, Check, Mail, Share2 } from 'lucide-react';
import {
    analyzeTextStream,
    listAvailableModels,
    PROMPT_WORDS_LONG,
    buildDeepReadPrompt,
    MODEL_DISPLAY_NAMES,
} from './lib/gemini';
import './App.css';

type FunctionType = 'words' | 'tutor';
type AppView = 'main' | 'settings' | 'result';

// ==========================================
// Deep Read 構造化データ型
// ==========================================
interface DeepSectionItem {
    type: 'section';
    title: string;
    para: string;
}

interface DeepSentenceItem {
    type: 'sentence';
    original: string;
    chunkedEn: string;
    translation: string;
    chunkedJa: string;
    tips: string;
    vocabulary?: string;
}

type DeepReadItem = DeepSectionItem | DeepSentenceItem;

// ==========================================
// ストリームパーサー（メモ化で画面揺れ防止）
// ==========================================
const memoizedDeepBlocks = new Map<string, DeepReadItem>();

function parseSection(content: string): DeepSectionItem {
    const titleMatch = content.match(/<title>([\s\S]*?)<\/title>/);
    const paraMatch = content.match(/<para>([\s\S]*?)<\/para>/);
    const title = titleMatch ? titleMatch[1].trim() : content.match(/<title>([\s\S]*?)(?:<\/title>|<para>|\[SECTION_END\]|$)/)?.[1]?.trim() || '';
    const para = paraMatch ? paraMatch[1].trim() : content.match(/<para>([\s\S]*?)(?:<\/para>|\[SECTION_END\]|$)/)?.[1]?.trim() || '';
    return { type: 'section', title, para };
}

function parseSentence(content: string): DeepSentenceItem {
    const original = content.match(/<original>([\s\S]*?)(?:<\/original>|$)/)?.[1]?.trim() || '';
    const chunkedEn = content.match(/<chunked_en>([\s\S]*?)(?:<\/chunked_en>|$)/)?.[1]?.trim() || '';
    const translation = content.match(/<(?:translation|natural)>([\s\S]*?)(?:<\/(?:translation|natural)>|$)/)?.[1]?.trim() || '';
    const chunkedJa = content.match(/<chunked_ja>([\s\S]*?)(?:<\/chunked_ja>|$)/)?.[1]?.trim() || '';
    const tips = content.match(/<tips>([\s\S]*?)(?:<\/tips>|\[BLOCK_END\]|$)/)?.[1]?.trim() || '';
    const vocabulary = content.match(/<vocabulary>([\s\S]*?)(?:<\/vocabulary>|$)/)?.[1]?.trim() || '';
    return { type: 'sentence', original, chunkedEn, translation, chunkedJa, tips, vocabulary };
}

function parseDeepReadStream(text: string): DeepReadItem[] {
    const items: DeepReadItem[] = [];
    const parts = text.split(/(\[SECTION_START\]|\[BLOCK_START\])/);

    for (let i = 1; i < parts.length; i += 2) {
        const marker = parts[i];
        const content = parts[i + 1] || '';
        const isLast = i >= parts.length - 2;
        const cacheKey = marker + content;

        if (!isLast && memoizedDeepBlocks.has(cacheKey)) {
            items.push(memoizedDeepBlocks.get(cacheKey)!);
            continue;
        }

        const item: DeepReadItem = marker === '[SECTION_START]'
            ? parseSection(content)
            : parseSentence(content);

        if (!isLast) memoizedDeepBlocks.set(cacheKey, item);
        items.push(item);
    }

    return items;
}

// ==========================================
// Words: Markdownフォーマッタ
// ==========================================
function escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function applyInline(html: string): string {
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`(.+?)`/g, '<code class="inline-code">$1</code>');
    return html;
}

function formatMarkdown(text: string): string {
    if (text.includes('error-display')) return text;
    const lines = text.split('\n');
    const parts: string[] = [];
    let inBlockquote = false;
    let blockquoteLines: string[] = [];

    for (const line of lines) {
        if (line.trim() === '---' || line.trim() === '***') {
            if (inBlockquote) { parts.push(renderBQ(blockquoteLines)); blockquoteLines = []; inBlockquote = false; }
            parts.push('<hr class="result-hr">');
            continue;
        }
        if (line.startsWith('> ')) { inBlockquote = true; blockquoteLines.push(line.slice(2)); continue; }
        else if (inBlockquote) { parts.push(renderBQ(blockquoteLines)); blockquoteLines = []; inBlockquote = false; }

        if (line.trim() === '') {
            if (parts.length > 0 && !parts[parts.length - 1].includes('spacer')) parts.push('<div class="spacer"></div>');
            continue;
        }
        const sectionMatch = line.match(/^【(.+?)】$/);
        if (sectionMatch) { parts.push(`<div class="section-header">${escapeHtml(sectionMatch[1])}</div>`); continue; }
        if (line.startsWith('## ')) { parts.push(`<h2 class="result-h2">${applyInline(escapeHtml(line.slice(3)))}</h2>`); continue; }
        if (line.match(/^- /)) { parts.push(`<div class="bullet-item"><span class="bullet-dot">•</span><span>${applyInline(escapeHtml(line.slice(2)))}</span></div>`); continue; }
        if (line.startsWith('・')) { parts.push(`<div class="bullet-item"><span class="bullet-dot">•</span><span>${applyInline(escapeHtml(line.slice(1)))}</span></div>`); continue; }
        if (line.startsWith('▶')) { parts.push(`<div class="usage-example"><span class="usage-mark">▶</span><span>${applyInline(escapeHtml(line.slice(1).trim()))}</span></div>`); continue; }
        if (line.startsWith('→')) { parts.push(`<div class="arrow-line">→ ${applyInline(escapeHtml(line.slice(1).trim()))}</div>`); continue; }
        const numMatch = line.match(/^(\d+)\.\s+(.+)$/);
        if (numMatch) { parts.push(`<div class="numbered-item"><span class="num">${numMatch[1]}.</span><span>${applyInline(escapeHtml(numMatch[2]))}</span></div>`); continue; }
        parts.push(`<div class="text-line">${applyInline(escapeHtml(line))}</div>`);
    }
    if (inBlockquote && blockquoteLines.length > 0) parts.push(renderBQ(blockquoteLines));
    return parts.join('');
}

function renderBQ(lines: string[]): string {
    return `<blockquote class="result-blockquote">${lines.map(t => `<div>${escapeHtml(t)}</div>`).join('')}</blockquote>`;
}

function buildDeepReadHtmlForCopy(items: DeepReadItem[], showChunks: boolean): string {
    const parts: string[] = [];
    for (const item of items) {
        if (item.type === 'section') {
            if (item.title) parts.push(`<div style="font-weight:700;color:#222;margin:16px 0 4px;">${escapeHtml(item.title)}</div>`);
            if (item.para) parts.push(`<div style="color:#555;margin:4px 0 12px;line-height:1.6;">${escapeHtml(item.para)}</div>`);
        } else {
            const displayEn = showChunks && item.chunkedEn ? item.chunkedEn : item.original;
            const rawJa = showChunks ? (item.chunkedJa || item.translation) : item.translation;
            let transText = rawJa;
            let vocabLines: string[] = [];
            if (item.vocabulary && item.vocabulary.trim() !== '' && item.vocabulary.trim() !== 'なし') {
                vocabLines = item.vocabulary.split('\n').filter(l => l.trim() !== '' && l.trim() !== 'なし');
            } else {
                const lines = transText.split('\n').filter(l => l.trim() !== '');
                vocabLines = lines.filter(l => l.trim().startsWith('・') || l.trim().startsWith('•'));
                transText = lines.filter(l => !l.trim().startsWith('・') && !l.trim().startsWith('•')).join('\n');
            }
            parts.push(`<div style="border-left:3px solid #4a9eff;padding:4px 12px;margin:8px 0;color:#1a3a5c;">${escapeHtml(displayEn)}</div>`);
            parts.push(`<div style="color:#333;margin:4px 0;">${escapeHtml(transText)}</div>`);
            if (vocabLines.length > 0) {
                const vocabItems = vocabLines.map(line => {
                    const text = line.trim().replace(/^[・•\-*]\s*/, '');
                    return `<div style="color:#333;padding:2px 0;font-size:0.9em;">・${escapeHtml(text)}</div>`;
                }).join('');
                parts.push(`<div style="margin:8px 0;padding:6px 10px;background:#f5f5f5;border-radius:4px;"><div style="font-size:0.75em;font-weight:600;color:#555;margin-bottom:4px;">単語／イディオム</div>${vocabItems}</div>`);
            }
            if (item.tips && item.tips.trim() !== '' && item.tips.trim() !== 'なし') {
                parts.push(`<div style="margin:6px 0;"><div style="font-size:0.75em;font-weight:600;color:#555;margin-bottom:2px;">Tips</div><div style="color:#555;font-size:0.9em;">${escapeHtml(item.tips).replace(/\n/g, '<br>')}</div></div>`);
            }
            parts.push('<hr style="border:none;border-top:1px solid #e0e0e0;margin:12px 0;">');
        }
    }
    return `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Sans',sans-serif;line-height:1.7;color:#333;">${parts.join('')}</div>`;
}

function shortenWordsResult(longResult: string): string {
    const lines = longResult.split('\n');
    const kept: string[] = [];
    let inSection = '';
    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.match(/^[-*_⸻—=]{3,}$/)) continue;
        if (line.startsWith('## ')) { kept.push(line); continue; }
        if (line.match(/^【発音記号】/)) { inSection = 'pron'; kept.push(line); continue; }
        if (line.match(/^【意味】/) || line.match(/^【英和辞典】/)) { inSection = 'ja'; kept.push(line); continue; }
        if (line.match(/^【英英辞典】|^【コロケーション|^【シチュエーション|^【同意語|^【覚えておくべき/)) { inSection = 'skip'; continue; }
        if (line.match(/^【/)) { inSection = 'other'; kept.push(line); continue; }
        if (inSection === 'pron' || inSection === 'ja' || inSection === 'other' || inSection === '') kept.push(line);
    }
    return kept.join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

// ==========================================
// Deep Read 表示コンポーネント（memo化で画面揺れ防止）
// ==========================================

// 段落ブロック（チャンクボタン非対象・枠囲み）
const DeepSectionBlock = memo(({ item }: { item: DeepSectionItem }) => (
    <div className="dr-section-block">
        {item.title && <div className="dr-section-title">{item.title}</div>}
        {item.para && (
            <div className="dr-para-box">{item.para}</div>
        )}
    </div>
));

// 文ブロック（チャンクボタン対象）— WEB版と同じレイアウト順
const DeepSentenceBlock = memo(({ item, showChunks }: {
    item: DeepSentenceItem;
    showChunks: boolean;
}) => {
    const displayEn = showChunks && item.chunkedEn ? item.chunkedEn : item.original;
    const displayJa = showChunks ? (item.chunkedJa || item.translation) : item.translation;

    // 語彙行の抽出
    let vocabLines: string[] = [];
    let transText = displayJa;
    if (item.vocabulary && item.vocabulary !== 'なし') {
        vocabLines = item.vocabulary.split('\n').filter(l => l.trim() !== '' && l.trim() !== 'なし');
    } else {
        const lines = transText.split('\n').filter(l => l.trim() !== '');
        vocabLines = lines.filter(l => l.trim().startsWith('・') || l.trim().startsWith('•'));
        transText = lines.filter(l => !l.trim().startsWith('・') && !l.trim().startsWith('•')).join('\n');
    }

    // チャンクON/OFFでスラッシュ表示切り替え
    let formattedTrans = escapeHtml(transText);
    if (showChunks) {
        formattedTrans = formattedTrans.replace(/[／/]/g, '<span class="chunk-slash">／</span>');
    } else {
        formattedTrans = formattedTrans.replace(/[／/]/g, '');
    }

    return (
        <div className="dr-sentence-block">
            {/* ① 英文（WEB版と同じ blockquote） */}
            <blockquote className="dr-original-quote">
                <div className="tutor-line">
                    <span dangerouslySetInnerHTML={{
                        __html: escapeHtml(displayEn).replace(/[／/]/g, showChunks ? '<span class="chunk-slash">／</span>' : ' ')
                    }} />
                </div>
            </blockquote>
            {/* ② 日本語訳（WEB版と同じ text-line 直下） */}
            <div
                className={`text-line ${showChunks ? 'chunked-text' : 'natural-text'}`}
                dangerouslySetInnerHTML={{ __html: formattedTrans }}
            />
            {/* ③ 単語／イディオム解説 */}
            {vocabLines.length > 0 && (
                <div className="dr-vocab-section">
                    <div className="dr-label">単語／イディオム</div>
                    {vocabLines.map((line, idx) => (
                        <div key={idx} className="dr-vocab-item">{line.trim().replace(/^[・•\-*]\s*/, '')}</div>
                    ))}
                </div>
            )}
            {/* ④ Tips */}
            {item.tips && item.tips !== 'なし' && (
                <div className="dr-tips">
                    <div><span className="dr-label tips">Tips</span></div>
                    <div className="dr-tips-text">{item.tips}</div>
                </div>
            )}
            <hr className="result-hr" />
        </div>
    );
});

// ==========================================
// メインコンポーネント
// ==========================================
function App() {
    const [view, setView] = useState<AppView>('main');
    const [sourceText, setSourceText] = useState('');
    const [resultContent, setResultContent] = useState('');
    const [deepReadItems, setDeepReadItems] = useState<DeepReadItem[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isDone, setIsDone] = useState(false);
    const [activeFunction, setActiveFunction] = useState<FunctionType | null>(null);
    const [errorMessage, setErrorMessage] = useState('');

    // Settings
    const [apiKey, setApiKey] = useState(localStorage.getItem('deepread_apikey') || '');
    const [model, setModel] = useState(localStorage.getItem('deepread_model') || 'gemini-2.5-flash');
    const [shareEmail, setShareEmail] = useState(localStorage.getItem('deepread_share_email') || '');
    const [showApiKey, setShowApiKey] = useState(false);
    const [settingsStatus, setSettingsStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    // CEFR level
    const [cefrLevel, setCefrLevel] = useState(localStorage.getItem('deepread_cefr') || 'B1');

    // Model selection
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [loadingModels, setLoadingModels] = useState(false);
    const fetchingModelsRef = useRef(false);
    const [modelDropdownOpen, setModelDropdownOpen] = useState(false);

    // Chunk表示切替（tutorモード）
    const [showChunks, setShowChunks] = useState(false);

    // Words: Long/Short切替
    const [wordsMode, setWordsMode] = useState<'long' | 'short'>('long');
    const [wordsFullResult, setWordsFullResult] = useState('');

    const [copySuccess, setCopySuccess] = useState(false);

    const resultRef = useRef<HTMLDivElement>(null);
    const activeRequestIdRef = useRef<number>(0);

    // 初期化
    useEffect(() => {
        if (!localStorage.getItem('deepread_apikey')) {
            setView('settings');
        } else {
            fetchModels();
        }
        const handleClickOutside = (e: MouseEvent) => {
            if (modelDropdownOpen) {
                const container = document.getElementById('model-selector-container');
                if (container && !container.contains(e.target as Node)) {
                    setModelDropdownOpen(false);
                }
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => { document.removeEventListener('mousedown', handleClickOutside); };
    }, [modelDropdownOpen]);

    // モデル一覧取得
    const fetchModels = useCallback(async () => {
        const key = localStorage.getItem('deepread_apikey');
        if (!key || fetchingModelsRef.current) return;
        fetchingModelsRef.current = true;
        setLoadingModels(true);
        try {
            const models = await listAvailableModels(key);
            setAvailableModels(models);
            if (models.length > 0 && !models.includes(model)) {
                setModel(models[0]);
                localStorage.setItem('deepread_model', models[0]);
            }
        } catch (err) {
            console.error('Failed to fetch models:', err);
        } finally {
            fetchingModelsRef.current = false;
            setLoadingModels(false);
        }
    }, [model]);

    useEffect(() => {
        if (view === 'main' && localStorage.getItem('deepread_apikey')) {
            fetchModels();
        }
    }, [view, fetchModels]);

    // ==========================================
    // API呼び出し
    // ==========================================
    const handleExecute = async (type: FunctionType) => {
        if (!sourceText.trim()) return;
        const storedKey = localStorage.getItem('deepread_apikey');
        if (!storedKey) { setView('settings'); return; }

        memoizedDeepBlocks.clear();

        const activeModel = model;
        setActiveFunction(type);
        setView('result');
        setResultContent('');
        setDeepReadItems([]);
        setWordsFullResult('');
        setWordsMode('long');
        setShowChunks(false);
        setIsLoading(true);
        setIsDone(false);
        setErrorMessage('');

        const requestId = Date.now();
        activeRequestIdRef.current = requestId;

        const promptMap: Record<FunctionType, string> = {
            words: PROMPT_WORDS_LONG,
            tutor: buildDeepReadPrompt(),
        };

        try {
            const finalResult = await analyzeTextStream(
                storedKey,
                sourceText,
                promptMap[type],
                activeModel,
                (accumulated) => {
                    if (activeRequestIdRef.current !== requestId) return false;
                    if (type === 'tutor') {
                        setDeepReadItems(parseDeepReadStream(accumulated));
                    } else {
                        setResultContent(accumulated);
                    }
                }
            );

            if (activeRequestIdRef.current !== requestId) return;

            if (type === 'tutor') {
                setDeepReadItems(parseDeepReadStream(finalResult));
            } else {
                const cleaned = finalResult.replace(
                    /^(はい、承知いたしました。|承知いたしました。|かしこまりました。)[^\n]*\n*/i, ''
                );
                setResultContent(cleaned);
                setWordsFullResult(cleaned);
            }

            setIsDone(true);
        } catch (err: any) {
            if (activeRequestIdRef.current !== requestId) return;
            console.error('handleExecute Error:', err);
            const msg = err?.message || '不明なエラーが発生しました';
            const isQuotaError = msg.includes('[RESOURCE_EXHAUSTED]') || msg.includes('429') || msg.includes('quota');
            const isAuthError = msg.includes('[AUTH_ERROR]') || msg.includes('API key') || msg.includes('401') || msg.includes('403');
            const isOverloaded = msg.includes('[OVERLOADED]') || msg.includes('503');

            if (isAuthError) {
                setErrorMessage('APIキーが無効です。設定画面でAPIキーを確認してください。');
            } else if (isQuotaError) {
                setErrorMessage(`${MODEL_DISPLAY_NAMES[activeModel] || activeModel} の使用回数が上限に達しました。別のモデルを選択してください。`);
            } else if (isOverloaded) {
                setErrorMessage(`${MODEL_DISPLAY_NAMES[activeModel] || activeModel} は現在混雑しています。しばらく待つか別のモデルをお試しください。`);
            } else if (msg.includes('[SAFETY_ERROR]')) {
                setErrorMessage('安全フィルターにより内容がブロックされました。入力を調整してください。');
            } else if (msg.includes('404') || msg.includes('not found')) {
                setErrorMessage(`モデル「${MODEL_DISPLAY_NAMES[activeModel] || activeModel}」が見つかりません。別のモデルを選択してください。`);
            } else {
                setErrorMessage('エラーが発生しました。しばらく待ってからもう一度お試しください。');
            }
        } finally {
            setIsLoading(false);
        }
    };

    // ==========================================
    // 設定保存
    // ==========================================
    const handleSaveSettings = () => {
        if (!apiKey.trim()) {
            setSettingsStatus({ type: 'error', text: 'APIキーを入力してください' });
            return;
        }
        localStorage.setItem('deepread_apikey', apiKey.trim());
        localStorage.setItem('deepread_model', model);
        localStorage.setItem('deepread_share_email', shareEmail.trim());
        setSettingsStatus({ type: 'success', text: '✓ 設定を保存しました！' });
        setTimeout(() => {
            setSettingsStatus(null);
            setView('main');
        }, 1500);
    };

    // ブラウザ戻るボタン対策
    useEffect(() => {
        if (view !== 'main') {
            window.history.pushState(null, '', window.location.href);
        }
    }, [view]);

    useEffect(() => {
        const handlePopState = () => {
            activeRequestIdRef.current = 0;
            setView('main');
            setIsLoading(false);
            setSourceText('');
        };
        window.addEventListener('popstate', handlePopState);
        return () => { window.removeEventListener('popstate', handlePopState); };
    }, []);

    // ==========================================
    // コピー / シェア
    // ==========================================
    const getResultText = useCallback(() => {
        if (activeFunction === 'tutor') {
            return deepReadItems.map(item => {
                if (item.type === 'section') {
                    return [item.title, item.para].filter(Boolean).join('\n');
                } else {
                    const en = showChunks ? (item.chunkedEn || item.original) : item.original;
                    const rawJa = showChunks ? (item.chunkedJa || item.translation) : item.translation;
                    // 表示側（DeepSentenceBlock）と同じロジックで語彙を抽出
                    let transText = rawJa;
                    let vocabText = '';
                    if (item.vocabulary && item.vocabulary.trim() !== '' && item.vocabulary.trim() !== 'なし') {
                        vocabText = '【単語／イディオム】\n' + item.vocabulary;
                    } else {
                        // vocabularyが空のとき、翻訳文から・行を語彙として抽出（表示と同じ処理）
                        const lines = transText.split('\n').filter(l => l.trim() !== '');
                        const vocabLines = lines.filter(l => l.trim().startsWith('・') || l.trim().startsWith('•'));
                        if (vocabLines.length > 0) {
                            vocabText = '【単語／イディオム】\n' + vocabLines.join('\n');
                            transText = lines.filter(l => !l.trim().startsWith('・') && !l.trim().startsWith('•')).join('\n');
                        }
                    }
                    // tipsが「なし」の場合は除外
                    const tipsText = (item.tips && item.tips.trim() !== 'なし') ? item.tips : '';
                    return [en, transText, vocabText, tipsText].filter(Boolean).join('\n');
                }
            }).join('\n\n');
        }
        if (activeFunction === 'words' && wordsMode === 'short' && wordsFullResult) {
            return shortenWordsResult(wordsFullResult);
        }
        return resultContent;
    }, [activeFunction, deepReadItems, showChunks, wordsMode, wordsFullResult, resultContent]);

    const copyRichText = useCallback(async () => {
        const plainText = getResultText();
        if (!plainText) return false;
        let htmlContent: string;
        if (activeFunction === 'tutor') {
            htmlContent = buildDeepReadHtmlForCopy(deepReadItems, showChunks);
        } else {
            const resultEl = resultRef.current;
            if (!resultEl) return false;
            const html = resultEl.innerHTML;
            const normalizedHtml = html
                .replace(/<blockquote[^>]*>/g, '<div style="margin:0;padding:0;border:none;">')
                .replace(/<\/blockquote>/g, '</div>');
            htmlContent = `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Sans',sans-serif;line-height:1.7;">${normalizedHtml}</div>`;
        }
        try {
            const htmlBlob = new Blob([htmlContent], { type: 'text/html' });
            const textBlob = new Blob([plainText], { type: 'text/plain' });
            await navigator.clipboard.write([
                new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob })
            ]);
            return true;
        } catch {
            await navigator.clipboard.writeText(plainText);
            return true;
        }
    }, [activeFunction, deepReadItems, showChunks, getResultText]);

    const handleMemo = useCallback(async () => {
        const text = getResultText();
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
            setCopySuccess(true);
            setTimeout(() => setCopySuccess(false), 2000);
        } catch { /* ignore */ }
    }, [getResultText]);

    const handleOpenGmail = useCallback(async () => {
        const text = getResultText();
        if (!text) return;
        const to = localStorage.getItem('deepread_share_email') || '';
        const subject = encodeURIComponent('Deep Read 解析結果');
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        try {
            await copyRichText();
            setCopySuccess(true);
            setTimeout(() => setCopySuccess(false), 2000);
        } catch (_) { /* ignore */ }
        if (isMobile) {
            window.location.href = `mailto:${to}?subject=${subject}`;
        } else {
            window.open(`https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(to)}&su=${subject}`, '_blank');
        }
    }, [getResultText, copyRichText]);

    const handleNativeShare = useCallback(async () => {
        const text = getResultText();
        if (!text || !navigator.share) return;
        try { await navigator.clipboard.writeText(text); } catch { /* ignore */ }
        try { await navigator.share({ title: 'Deep Read 解析結果', text }); } catch (_) { /* cancel */ }
    }, [getResultText]);

    const canNativeShare = typeof navigator !== 'undefined' && !!navigator.share;

    // ==========================================
    // 表示用ヘルパー
    // ==========================================
    const displayModel = MODEL_DISPLAY_NAMES[model] || model;

    const functionLabel: Record<FunctionType, string> = {
        words: 'English Words',
        tutor: 'Deep Read',
    };

    const CEFR_OPTIONS = [
        { value: 'A1', label: 'A1' },
        { value: 'A2', label: 'A2' },
        { value: 'B1', label: 'B1' },
        { value: 'B2', label: 'B2' },
        { value: 'C1', label: 'C1' },
        { value: 'C2', label: 'C2' },
    ];

    // ==========================================
    // 設定画面
    // ==========================================
    if (view === 'settings') {
        return (
            <div className="deepread-container">
                <header className="deepread-header">
                    <div className="header-brand">
                        <div className="logo-icon"><BookMarked size={22} /></div>
                        <h1>Deep Read</h1>
                    </div>
                    {localStorage.getItem('deepread_apikey') && (
                        <button className="back-btn" onClick={() => { setView('main'); setIsLoading(false); setErrorMessage(''); }}>
                            <ArrowLeft size={18} /><span>戻る</span>
                        </button>
                    )}
                </header>
                <main className="deepread-main">
                    <div className="canvas-card settings-card">
                        <div className="canvas-header">
                            <h2 style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <Key size={15} /> API設定
                            </h2>
                        </div>
                        <div className="settings-form">
                            <div className="form-group">
                                <label>Gemini API Key</label>
                                <p className="form-hint">
                                    <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a> でAPIキーを取得して入力してください。
                                </p>
                                <div className="input-row">
                                    <input
                                        type={showApiKey ? 'text' : 'password'}
                                        value={apiKey}
                                        onChange={(e) => setApiKey(e.target.value)}
                                        placeholder="AIza..."
                                        className="api-input"
                                    />
                                    <button type="button" className="toggle-btn" onClick={() => setShowApiKey(!showApiKey)}>
                                        {showApiKey ? '隠す' : '表示'}
                                    </button>
                                </div>
                            </div>
                            <div className="form-group">
                                <label>送信先メールアドレス（任意）</label>
                                <p className="form-hint">Gmailボタンを使う場合、宛先として自動入力されます。</p>
                                <input
                                    type="email"
                                    value={shareEmail}
                                    onChange={(e) => setShareEmail(e.target.value)}
                                    placeholder="example@gmail.com"
                                    className="api-input"
                                />
                            </div>
                            <button className="save-btn" onClick={handleSaveSettings}>保存して開始</button>
                            {settingsStatus && (
                                <div className={`settings-status ${settingsStatus.type}`}>{settingsStatus.text}</div>
                            )}
                        </div>
                    </div>
                </main>
                <footer className="deepread-footer">
                    <span className="footer-glow">DEEP READ • ENGLISH READING & WRITING COACH</span>
                </footer>
            </div>
        );
    }

    // ==========================================
    // 結果画面
    // ==========================================
    if (view === 'result' && activeFunction) {
        const displayContent = activeFunction === 'words' && wordsMode === 'short' && wordsFullResult
            ? shortenWordsResult(wordsFullResult)
            : resultContent;

        return (
            <div className="deepread-container">
                <header className="deepread-header result-header">
                    <div className="header-brand">
                        <button className="back-btn" onClick={() => {
                            activeRequestIdRef.current = 0; // 実行中のストリームを中断
                            setView('main');
                            setIsLoading(false);
                            setSourceText('');
                        }}>
                            <ArrowLeft size={18} />
                        </button>
                        <div className="logo-icon small"><BookMarked size={16} /></div>
                        <h1>{functionLabel[activeFunction]}</h1>
                        {isLoading && (
                            <div className="loading-display mini" style={{ marginLeft: '12px' }}>
                                <div className="spinner small" />
                                <span>解析中...</span>
                            </div>
                        )}
                    </div>
                    <div className="header-right">
                        <div className="header-btn-group">
                            {/* チャンクボタン（tutorモードのみ） */}
                            {activeFunction === 'tutor' && (
                                <button
                                    className={`mode-toggle-btn ${showChunks ? 'active' : ''}`}
                                    onClick={() => setShowChunks(!showChunks)}
                                >
                                    ／ Chunk {showChunks ? 'ON' : 'OFF'}
                                </button>
                            )}
                            {/* Long/Short（wordsモードのみ） */}
                            {activeFunction === 'words' && isDone && (
                                <button
                                    className={`mode-toggle-btn ${wordsMode === 'short' ? 'active' : ''}`}
                                    onClick={() => setWordsMode(wordsMode === 'long' ? 'short' : 'long')}
                                >
                                    {wordsMode === 'long' ? '📖 Long' : '📋 Short'}
                                </button>
                            )}
                            {isDone && !errorMessage && (
                                <button
                                    className={`mode-toggle-btn ${copySuccess ? 'active' : ''}`}
                                    onClick={handleMemo}
                                    title="テキストをメモにコピー"
                                >
                                    {copySuccess ? <Check size={14} /> : <Copy size={14} />}
                                    <span>{copySuccess ? 'コピー済' : 'メモ'}</span>
                                </button>
                            )}
                        </div>
                        {isDone && !errorMessage && (
                            <div className="header-btn-group">
                                <button className="mode-toggle-btn" onClick={handleOpenGmail} title="Gmailで送る">
                                    <Mail size={14} /><span>Gmail</span>
                                </button>
                                {canNativeShare && (
                                    <button className="mode-toggle-btn" onClick={handleNativeShare} title="共有">
                                        <Share2 size={14} /><span>共有</span>
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </header>

                <div className="source-preview">
                    {sourceText.length > 120 ? sourceText.substring(0, 120) + '…' : sourceText}
                </div>

                <main className="result-main" ref={resultRef}>
                    {errorMessage ? (
                        <div className="error-display">
                            <div className="error-icon">⚠️</div>
                            <div className="error-text">{errorMessage}</div>
                            <button className="retry-btn" onClick={() => handleExecute(activeFunction)}>再試行</button>
                        </div>
                    ) : activeFunction === 'tutor' ? (
                        /* Deep Read 結果（構造化ブロック） */
                        <div className="dr-result">
                            {deepReadItems.map((item, i) =>
                                item.type === 'section'
                                    ? <DeepSectionBlock key={i} item={item} />
                                    : <DeepSentenceBlock key={i} item={item} showChunks={showChunks} />
                            )}
                        </div>
                    ) : displayContent ? (
                        /* English Words 結果（Markdown） */
                        <div className="result-content" dangerouslySetInnerHTML={{
                            __html: formatMarkdown(displayContent)
                        }} />
                    ) : isDone ? (
                        <div className="error-display">
                            <div className="error-icon">⚠️</div>
                            <div className="error-text">結果を取得できませんでした。モデルを変更するか、再試行してください。</div>
                            <button className="retry-btn" onClick={() => handleExecute(activeFunction)}>再試行</button>
                        </div>
                    ) : null}
                </main>

                <footer className="deepread-footer">
                    <span className="footer-glow">
                        DEEP READ • {functionLabel[activeFunction].toUpperCase()} • {displayModel}
                        {activeFunction === 'tutor' ? ` • CEFR ${cefrLevel}` : ''}
                    </span>
                </footer>
            </div>
        );
    }

    // ==========================================
    // メイン画面
    // ==========================================
    return (
        <div className="deepread-container">
            <header className="deepread-header">
                <div className="header-brand">
                    <div className="logo-icon"><BookMarked size={22} /></div>
                    <h1>Deep Read</h1>
                </div>
                <div className="header-right">
                    <button className="settings-btn" onClick={() => setView('settings')} title="設定">
                        <Settings size={18} />
                    </button>
                </div>
            </header>

            <main className="deepread-main">
                <div className="canvas-card">
                    <div className="canvas-header">
                        <h2>INPUT TEXT</h2>
                        <div className="canvas-header-right">
                            {sourceText && (
                                <button className="clear-btn" onClick={() => setSourceText('')}>Clear</button>
                            )}

                            {/* CEFRレベル選択 */}
                            <div className="cefr-selector-wrap">
                                <select
                                    className="cefr-selector"
                                    value={cefrLevel}
                                    onChange={(e) => {
                                        setCefrLevel(e.target.value);
                                        localStorage.setItem('deepread_cefr', e.target.value);
                                    }}
                                >
                                    {CEFR_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                            </div>

                            {/* モデル選択ドロップダウン */}
                            <div className="model-selector" id="model-selector-container">
                                <button
                                    className="model-selector-btn"
                                    onClick={() => {
                                        setModelDropdownOpen(!modelDropdownOpen);
                                        if (!modelDropdownOpen) fetchModels();
                                    }}
                                >
                                    <span className="model-name">{displayModel}</span>
                                    <ChevronDown size={14} className={modelDropdownOpen ? 'rotate-180' : ''} />
                                </button>
                                {modelDropdownOpen && (
                                    <div className="model-dropdown">
                                        <div className="model-dropdown-header">
                                            <span>Select AI Model</span>
                                        </div>
                                        {loadingModels ? (
                                            <div className="model-dropdown-loading">更新中...</div>
                                        ) : (
                                            availableModels.map(m => (
                                                <button
                                                    key={m}
                                                    className={`model-option ${m === model ? 'active' : ''}`}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setModel(m);
                                                        localStorage.setItem('deepread_model', m);
                                                        setModelDropdownOpen(false);
                                                    }}
                                                >
                                                    <span className="model-option-name">{MODEL_DISPLAY_NAMES[m] || m}</span>
                                                </button>
                                            ))
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="input-workspace">
                        <textarea
                            className="main-textarea"
                            placeholder=""
                            value={sourceText}
                            onChange={(e) => setSourceText(e.target.value)}
                        />
                    </div>

                    <div className="action-panel">
                        <h3 className="panel-title">CHOOSE FUNCTION</h3>
                        <div className="button-group">
                            <button
                                className="action-btn words-btn"
                                onClick={() => handleExecute('words')}
                                disabled={!sourceText.trim() || isLoading}
                            >
                                <Search size={20} />
                                <span>English Words<br /><small>単語と熟語の解説</small></span>
                            </button>
                            <button
                                className="action-btn tutor-btn"
                                onClick={() => handleExecute('tutor')}
                                disabled={!sourceText.trim() || isLoading}
                            >
                                <BookOpen size={20} />
                                <span>Deep Read<br /><small>Reading &amp; Writing 解析</small></span>
                            </button>
                        </div>
                    </div>

                    {errorMessage && view === 'main' && (
                        <div className="error-toast">⚠️ {errorMessage}</div>
                    )}
                </div>
            </main>

            <footer className="deepread-footer">
                <span className="footer-glow">DEEP READ • ENGLISH READING & WRITING COACH</span>
            </footer>
        </div>
    );
}

export default App;
