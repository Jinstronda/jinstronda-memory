export interface AtomicFact {
  id: string
  content: string
  sessionId: string
  date?: string
  eventDate?: string
  factIndex: number
  embedding: number[]
}

export interface FactSearchResult {
  content: string
  score: number
  sessionId: string
  date?: string
  eventDate?: string
  factIndex: number
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0

  let dot = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB)
  if (denominator === 0) return 0

  return dot / denominator
}

export class InMemoryFactStore {
  private containers: Map<string, Map<string, AtomicFact>> = new Map()

  private getContainer(containerTag: string): Map<string, AtomicFact> {
    if (!this.containers.has(containerTag)) {
      this.containers.set(containerTag, new Map())
    }
    return this.containers.get(containerTag)!
  }

  addFacts(containerTag: string, facts: AtomicFact[]): void {
    const container = this.getContainer(containerTag)
    for (const fact of facts) {
      container.set(fact.id, fact)
    }
  }

  search(containerTag: string, queryEmbedding: number[], limit: number): FactSearchResult[] {
    const container = this.containers.get(containerTag)
    if (!container || container.size === 0) return []

    const scored: FactSearchResult[] = []
    for (const fact of container.values()) {
      const score = cosineSimilarity(queryEmbedding, fact.embedding)
      scored.push({
        content: fact.content,
        score,
        sessionId: fact.sessionId,
        date: fact.date,
        eventDate: fact.eventDate,
        factIndex: fact.factIndex,
      })
    }

    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, limit)
  }

  save(containerTag: string): { facts: AtomicFact[] } | null {
    const container = this.containers.get(containerTag)
    if (!container) return null
    return { facts: [...container.values()] }
  }

  load(containerTag: string, data: { facts: AtomicFact[] }): void {
    const container = new Map<string, AtomicFact>()
    for (const fact of data.facts) {
      container.set(fact.id, fact)
    }
    this.containers.set(containerTag, container)
  }

  hasData(containerTag: string): boolean {
    return (this.containers.get(containerTag)?.size || 0) > 0
  }

  clear(containerTag: string): void {
    this.containers.delete(containerTag)
  }

  getFactCount(containerTag: string): number {
    return this.containers.get(containerTag)?.size || 0
  }
}
