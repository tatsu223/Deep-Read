import { GoogleGenerativeAI } from "@google/generative-ai";

// ==========================================
// テキスト出力モデル
// ==========================================
export const TEXT_OUTPUT_MODELS = [
    'gemini-3-flash-preview',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
];

export const MODEL_DISPLAY_NAMES: Record<string, string> = {
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
};

// ==========================================
// English Words プロンプト
// ==========================================
export const PROMPT_WORDS_LONG = `
# Role
あなたはプロの英語講師および言語学者です。対象の英文から、重要・難解な英単語や熟語を全てピックアップし、以下の【出力フォーマット】を厳守して、その単語の深い理解を助ける解説を生成してください。

# Constraints
- 意味、語法、ニュアンスの違いを正確に説明すること。
- 例文は実用的で自然なものを作成すること。
- 日本語と英語を併記すること。
- 余計な挨拶やメタ発言は一切含めず、直接【出力フォーマット】から開始すること。

# 出力フォーマット

---
## [ここに単語]

【発音記号】
[IPA発音記号]

【意味】
・(品詞) [意味1]
・(品詞) [意味2]

【英和辞典】

([分類])
1. ([核心的な意味の要約])
・例: [英語例文]
（[日本語訳]）

【英英辞典】
・[英語での定義1]
→ [定義1の日本語要約]

【シチュエーション】
1. [カテゴリ名]
・[その状況での使い方の説明]
・例: [英語例文]
（[日本語訳]）

【覚えておくべきこと】
1. 語源
・[語源の由来と、それがどう現在の意味につながっているか]
2. フォーマルとカジュアルの使い分け
・[文脈による使い分け]

【同意語（類義語）】
・[類義語1] [発音記号] （[意味]）
→ [ターゲット単語とのニュアンスの違いを詳しく説明]

【コロケーション】
・[collocation 1] ([意味])
・[collocation 2] ([意味])
・[collocation 3] ([意味])
`;

// ==========================================
// Deep Read プロンプト（CEFRレベル別・構造化ブロック形式）
// ==========================================
export function buildDeepReadPrompt(): string {
    return `# Role
あなたは英文をチャンク（意味の塊）ごとに区切り、語順のまま理解させる指導を行うプロの英語講師です。

# Objective
入力された文章を段落ごとにグループ化し、各段落内の文を「一文（ピリオド・感嘆符・疑問符・句点単位）ごと」に分割して、以下の【出力フォーマット】で出力してください。
一文の中身は、1:自然な和訳、2:詳細なチャンク分け英文、3:英語の語順に従ったチャンク和訳、4:重要単語解説、5:文法Tips、を含めてください。

# Output Format
まず段落ごとに以下のセクションブロックを出力し、その直後に段落内の各文のブロックを出力してください：

[SECTION_START]
<title>第N段落</title>
<para>この段落の原文をそのまま記載する</para>
[SECTION_END]

次に、その段落内の各文について以下の形式で出力してください：

[BLOCK_START]
<original>元の英文（一文のみ）</original>
<natural>自然な日本語訳。注釈は一切含めず、純粋で自然な和訳のみを出力する。</natural>
<chunked_en>英文を意味の区切り「 ／ 」で分けたもの。チャンクの区切りは <chunked_ja> と必ず1:1で対応させること。</chunked_en>
<chunked_ja>英語の語順を厳守し「 ／ 」で区切った日本語訳。チャンク数は <chunked_en> と完全に一致させること。</chunked_ja>
<vocabulary>この文に含まれる重要単語・熟語・イディオムを箇条書きで解説する。各項目は「・単語/熟語：意味や補足説明」の形式で記載する。特に注目すべき単語がない場合は「なし」と記載。</vocabulary>
<tips>この文の文法・構文・表現に関するワンポイント解説を記載する。例：関係代名詞の用法、仮定法、時制の使い分け、コロケーションなど。特筆すべきものがない場合は「なし」と記載。</tips>
[BLOCK_END]

# Rules for Mastery
0. **【最重要】全文処理**: 入力テキストに含まれる全ての文を、一つも省略・統合・スキップせずに処理すること。入力がN文なら出力もN個の[BLOCK_START]〜[BLOCK_END]でなければならない。開始前に文数を数え、終了前に全文をカバーしたか確認すること。
1. **一文の定義**: ピリオド、クエスチョンマーク、感嘆符、または句点で終わるものを一文とします。
2. **チャンク分け**: 主節・従属節・長い前置詞句などの意味の切れ目で区切る。一単語ずつになるほど細かくせず、ネイティブがスピーキングでポーズを入れる自然な長さを意識すること。
3. **語順の遵守**: <chunked_ja> は英語が流れてくる順番を絶対に維持すること。戻り読みをせず、英語チャンクに対応する断片的な日本語を当てること。

4. **チャンク数の一致**: <chunked_en> と <chunked_ja> の「 ／ 」の個数は必ず同じにすること。1:1対応を崩さないこと。
5. **単語解説**: <vocabulary> には、学習者にとって有用な単語・熟語・イディオムを漏れなくピックアップし、簡潔に意味と補足を記載すること。
6. **Tips**: <tips> には、その文の理解に役立つ文法・構文・表現のポイントを1〜2文で簡潔に解説すること。
7. **タグの厳守**: 挨拶・メタ発言は一切不要。タグの構造のみを漏らさず出力すること。`;
}

// ==========================================
// API通信
// ==========================================
export async function analyzeTextStream(
    apiKey: string,
    text: string,
    systemPrompt: string,
    modelName: string,
    onChunk: (accumulated: string) => boolean | void
): Promise<string> {
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
        model: modelName,
        systemInstruction: systemPrompt
    }, { apiVersion: 'v1beta' });

    const result = await model.generateContentStream(text);
    let accumulated = "";

    try {
        for await (const chunk of result.stream) {
            try {
                const chunkText = chunk.text();
                if (chunkText) {
                    accumulated += chunkText;
                    if (onChunk(accumulated) === false) {
                        break;
                    }
                }
            } catch (chunkError: any) {
                console.warn(`[${modelName}] Chunk error:`, chunkError);
            }
        }
    } catch (streamError: any) {
        console.error(`[${modelName}] Stream error:`, streamError);
        const msg = streamError.message || "";
        let errorType = "[ERROR]";
        if (msg.includes("429") || msg.includes("quota") || msg.includes("exhausted")) errorType = "[RESOURCE_EXHAUSTED]";
        else if (msg.includes("503") || msg.includes("overloaded")) errorType = "[OVERLOADED]";
        else if (msg.includes("API key") || msg.includes("401") || msg.includes("403")) errorType = "[AUTH_ERROR]";
        else if (msg.includes("Safety") || msg.includes("block")) errorType = "[SAFETY_ERROR]";
        if (!accumulated) throw new Error(`${errorType} ${msg}`);
        console.warn(`${errorType} Stream stopped prematurely.`);
    }

    if (!accumulated) {
        throw new Error(`[ERROR] モデル "${modelName}" から応答が得られませんでした。`);
    }
    return accumulated;
}

export async function listAvailableModels(_apiKey: string): Promise<string[]> {
    return [...TEXT_OUTPUT_MODELS];
}
