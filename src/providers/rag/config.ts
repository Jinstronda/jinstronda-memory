export interface RAGConfig {
  // search pipeline
  enableReranker: boolean
  enableQueryRewrite: boolean
  enableGraphRAG: boolean
  enableAtomicFacts: boolean
  enableFactSessionBoost: boolean
  enableQueryDecomposition: boolean
  enableProfile: boolean

  // ingest pipeline
  enableExtraction: boolean
  enableEntityGraph: boolean
  enableProfileBuilding: boolean

  // tuning
  chunkSize: number
  chunkOverlap: number
  embeddingModel: string
  rerankOverfetch: number
  extractionConcurrency: number
  maxGlobalExtractions: number
  factSearchLimit: number
  factSessionBoost: number
  cacheDir: string
}

const defaults: RAGConfig = {
  enableReranker: false,
  enableQueryRewrite: false,
  enableGraphRAG: true,
  enableAtomicFacts: true,
  enableFactSessionBoost: true,
  enableQueryDecomposition: true,
  enableProfile: true,

  enableExtraction: true,
  enableEntityGraph: true,
  enableProfileBuilding: true,

  chunkSize: 1600,
  chunkOverlap: 320,
  embeddingModel: "text-embedding-3-large",
  rerankOverfetch: 10,
  extractionConcurrency: 10,
  maxGlobalExtractions: 300,
  factSearchLimit: 30,
  factSessionBoost: 0.1,
  cacheDir: "./data/cache/rag",
}

export function loadConfig(overrides?: Partial<RAGConfig>): RAGConfig {
  const envOverrides: Partial<RAGConfig> = {}

  const boolEnv = (key: string): boolean | undefined => {
    const v = process.env[key]
    if (v === undefined) return undefined
    return v === "1" || v === "true"
  }

  const numEnv = (key: string): number | undefined => {
    const v = process.env[key]
    if (v === undefined) return undefined
    const n = parseInt(v, 10)
    return isNaN(n) ? undefined : n
  }

  const strEnv = (key: string): string | undefined => process.env[key] || undefined

  if (boolEnv("RAG_RERANKER") !== undefined) envOverrides.enableReranker = boolEnv("RAG_RERANKER")
  if (boolEnv("RAG_QUERY_REWRITE") !== undefined) envOverrides.enableQueryRewrite = boolEnv("RAG_QUERY_REWRITE")
  if (boolEnv("RAG_GRAPH") !== undefined) envOverrides.enableGraphRAG = boolEnv("RAG_GRAPH")
  if (boolEnv("RAG_FACTS") !== undefined) envOverrides.enableAtomicFacts = boolEnv("RAG_FACTS")
  if (boolEnv("RAG_FACT_BOOST") !== undefined) envOverrides.enableFactSessionBoost = boolEnv("RAG_FACT_BOOST")
  if (boolEnv("RAG_DECOMPOSE") !== undefined) envOverrides.enableQueryDecomposition = boolEnv("RAG_DECOMPOSE")
  if (boolEnv("RAG_PROFILE") !== undefined) envOverrides.enableProfile = boolEnv("RAG_PROFILE")
  if (boolEnv("RAG_EXTRACTION") !== undefined) envOverrides.enableExtraction = boolEnv("RAG_EXTRACTION")
  if (boolEnv("RAG_ENTITY_GRAPH") !== undefined) envOverrides.enableEntityGraph = boolEnv("RAG_ENTITY_GRAPH")
  if (boolEnv("RAG_PROFILE_BUILD") !== undefined) envOverrides.enableProfileBuilding = boolEnv("RAG_PROFILE_BUILD")

  if (numEnv("RAG_CHUNK_SIZE") !== undefined) envOverrides.chunkSize = numEnv("RAG_CHUNK_SIZE")
  if (numEnv("RAG_CHUNK_OVERLAP") !== undefined) envOverrides.chunkOverlap = numEnv("RAG_CHUNK_OVERLAP")
  if (strEnv("RAG_EMBEDDING_MODEL") !== undefined) envOverrides.embeddingModel = strEnv("RAG_EMBEDDING_MODEL")
  if (numEnv("RAG_RERANK_OVERFETCH") !== undefined) envOverrides.rerankOverfetch = numEnv("RAG_RERANK_OVERFETCH")
  if (numEnv("RAG_EXTRACTION_CONCURRENCY") !== undefined) envOverrides.extractionConcurrency = numEnv("RAG_EXTRACTION_CONCURRENCY")
  if (numEnv("RAG_FACT_SEARCH_LIMIT") !== undefined) envOverrides.factSearchLimit = numEnv("RAG_FACT_SEARCH_LIMIT")
  if (strEnv("RAG_CACHE_DIR") !== undefined) envOverrides.cacheDir = strEnv("RAG_CACHE_DIR")

  return { ...defaults, ...envOverrides, ...overrides }
}
