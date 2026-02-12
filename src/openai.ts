/**
 * OpenAI API provider implementing ILLMSession.
 * Replaces node-llama-cpp for embeddings, query expansion, and reranking.
 */

import type {
  ILLMSession,
  EmbeddingResult,
  EmbedOptions,
  Queryable,
  QueryType,
  RerankDocument,
  RerankResult,
  RerankOptions,
  RerankDocumentResult,
} from "./llm.js";

const EMBED_MODEL = process.env.OPENAI_EMBED_MODEL || "text-embedding-3-large";
const EMBED_DIMENSIONS = parseInt(process.env.OPENAI_EMBED_DIMENSIONS || "3072", 10);
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5.2";
const FAST_MODEL = process.env.OPENAI_FAST_MODEL || "gpt-5.2-mini";
const BASE_URL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";

export function getOpenAIModelName(): string {
  return `openai/${EMBED_MODEL}`;
}

export class OpenAISession implements ILLMSession {
  private apiKey: string;
  private abortController: AbortController;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.abortController = new AbortController();
  }

  get isValid(): boolean {
    return !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  release(): void {
    this.abortController.abort();
  }

  // ---------------------------------------------------------------------------
  // Embeddings
  // ---------------------------------------------------------------------------

  async embed(text: string, _options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text]);
    return results[0] ?? null;
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const BATCH_SIZE = 100;
    const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      try {
        const embeddings = await this.callEmbedAPI(batch);
        for (let j = 0; j < embeddings.length; j++) {
          results[i + j] = {
            embedding: embeddings[j]!,
            model: getOpenAIModelName(),
          };
        }
      } catch (err) {
        // Batch failed — try individually
        for (let j = 0; j < batch.length; j++) {
          try {
            const single = await this.callEmbedAPI([batch[j]!]);
            results[i + j] = {
              embedding: single[0]!,
              model: getOpenAIModelName(),
            };
          } catch {
            results[i + j] = null;
          }
        }
      }
    }

    return results;
  }

  private async callEmbedAPI(inputs: string[], retries = 3): Promise<number[][]> {
    for (let attempt = 0; attempt <= retries; attempt++) {
      const response = await fetch(`${BASE_URL}/embeddings`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: EMBED_MODEL,
          input: inputs,
          dimensions: EMBED_DIMENSIONS,
        }),
        signal: this.abortController.signal,
      });

      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get("retry-after") || "1", 10);
        const delay = Math.max(retryAfter * 1000, 1000 * Math.pow(2, attempt));
        await new Promise(r => setTimeout(r, delay));
        continue;
      }

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`OpenAI embeddings API error ${response.status}: ${body}`);
      }

      const data = await response.json() as {
        data: { embedding: number[]; index: number }[];
      };

      return data.data
        .sort((a, b) => a.index - b.index)
        .map(d => d.embedding);
    }

    throw new Error("OpenAI embeddings API: max retries exceeded");
  }

  // ---------------------------------------------------------------------------
  // Query Expansion
  // ---------------------------------------------------------------------------

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;

    const systemPrompt = `You are a search query expander. Given a search query, generate alternative search queries.
Output ONLY lines in this exact format (no other text):
lex: <lexical search alternative>
vec: <semantic search alternative>
hyde: <hypothetical document snippet that would match>

Generate 2-3 lines. Each line must be a different type. Keep alternatives concise (under 15 words).`;

    const userPrompt = options.context
      ? `Context: ${options.context}\nQuery: ${query}`
      : `Query: ${query}`;

    try {
      const response = await fetch(`${BASE_URL}/chat/completions`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: FAST_MODEL,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          temperature: 0.7,
          max_completion_tokens: 300,
        }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`OpenAI chat API error ${response.status}`);
      }

      const data = await response.json() as {
        choices: { message: { content: string } }[];
      };

      const text = data.choices[0]?.message?.content?.trim() || "";
      return this.parseExpansion(text, query, includeLexical);
    } catch (error) {
      // Fallback
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }
  }

  private parseExpansion(text: string, query: string, includeLexical: boolean): Queryable[] {
    const lines = text.trim().split("\n");
    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

    const hasQueryTerm = (t: string): boolean => {
      const lower = t.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some(term => lower.includes(term));
    };

    const queryables: Queryable[] = lines.map(line => {
      const colonIdx = line.indexOf(":");
      if (colonIdx === -1) return null;
      const type = line.slice(0, colonIdx).trim();
      if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
      const content = line.slice(colonIdx + 1).trim();
      if (!hasQueryTerm(content)) return null;
      return { type: type as QueryType, text: content };
    }).filter((q): q is Queryable => q !== null);

    const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
    if (filtered.length > 0) return filtered;

    // Fallback
    const fallback: Queryable[] = [
      { type: 'hyde', text: `Information about ${query}` },
      { type: 'lex', text: query },
      { type: 'vec', text: query },
    ];
    return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
  }

  // ---------------------------------------------------------------------------
  // Reranking
  // ---------------------------------------------------------------------------

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: `openai/${FAST_MODEL}` };
    }

    // Build numbered passage list
    const passages = documents.map((doc, i) =>
      `[${i}] ${doc.text.slice(0, 500)}`
    ).join("\n\n");

    const systemPrompt = `You are a relevance scorer. Given a query and numbered passages, score each passage's relevance to the query from 0.0 to 1.0.
Respond ONLY with a JSON array: [{"index": 0, "score": 0.95}, ...]
Include ALL passage indices. Order by score descending.`;

    try {
      const response = await fetch(`${BASE_URL}/chat/completions`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: FAST_MODEL,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `Query: ${query}\n\nPassages:\n${passages}` },
          ],
          temperature: 0,
          max_completion_tokens: documents.length * 30,
          response_format: { type: "json_object" },
        }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`OpenAI chat API error ${response.status}`);
      }

      const data = await response.json() as {
        choices: { message: { content: string } }[];
      };

      const content = data.choices[0]?.message?.content?.trim() || "[]";
      const parsed = JSON.parse(content);

      // Handle both {results: [...]} and [...] formats
      const scores: { index: number; score: number }[] = Array.isArray(parsed)
        ? parsed
        : (parsed.results || parsed.scores || parsed.rankings || []);

      const results: RerankDocumentResult[] = scores
        .filter((s: any) => typeof s.index === 'number' && typeof s.score === 'number')
        .map((s: any) => ({
          file: documents[s.index]?.file || "",
          score: s.score,
          index: s.index,
        }))
        .sort((a: RerankDocumentResult, b: RerankDocumentResult) => b.score - a.score);

      // Add any missing documents with score 0
      const seen = new Set(results.map(r => r.index));
      for (let i = 0; i < documents.length; i++) {
        if (!seen.has(i)) {
          results.push({ file: documents[i]!.file, score: 0, index: i });
        }
      }

      return { results, model: `openai/${FAST_MODEL}` };
    } catch (error) {
      // Fallback: return documents in original order with uniform scores
      const results: RerankDocumentResult[] = documents.map((doc, i) => ({
        file: doc.file,
        score: 1 - (i / documents.length),
        index: i,
      }));
      return { results, model: `openai/${FAST_MODEL}` };
    }
  }

  // ---------------------------------------------------------------------------
  // Tool-calling chat (for agentic loops)
  // ---------------------------------------------------------------------------

  async chatWithTools(
    messages: any[],
    tools: any[]
  ): Promise<{
    content: string | null;
    tool_calls?: { id: string; function: { name: string; arguments: string }; type: string }[];
    finish_reason: string;
  }> {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: CHAT_MODEL,
        messages,
        tools,
        temperature: 0.3,
        max_completion_tokens: 2000,
      }),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI chat API error ${response.status}: ${body}`);
    }

    const data = await response.json() as {
      choices: {
        message: {
          content: string | null;
          tool_calls?: { id: string; function: { name: string; arguments: string }; type: string }[];
        };
        finish_reason: string;
      }[];
    };

    const choice = data.choices[0]!;
    return {
      content: choice.message.content,
      tool_calls: choice.message.tool_calls,
      finish_reason: choice.finish_reason,
    };
  }

  // ---------------------------------------------------------------------------
  // RAG Answer Generation (streaming)
  // ---------------------------------------------------------------------------

  async *answer(
    question: string,
    contexts: { file: string; text: string; score: number }[]
  ): AsyncGenerator<string> {
    const systemPrompt = `You are a helpful assistant that answers questions based on the provided context documents.
Use ONLY the information from the context to answer. If the context doesn't contain enough information, say so.
Be concise and specific. Cite sources by their filename when relevant.`;

    const contextBlock = contexts
      .map((ctx, i) => `--- ${ctx.file} (relevance: ${(ctx.score * 100).toFixed(0)}%) ---\n${ctx.text}`)
      .join("\n\n");

    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: CHAT_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Context:\n${contextBlock}\n\nQuestion: ${question}` },
        ],
        temperature: 0.3,
        max_completion_tokens: 1000,
        stream: true,
      }),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI chat API error ${response.status}: ${body}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") return;

        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) yield content;
        } catch {}
      }
    }
  }
}
