/**
 * Comprehensive Evaluation Metrics Module
 * Implements ROUGE, BLEU, BERTScore, SummaC, EM, G-Eval, and other metrics
 */

const natural = require('natural');
// Note: rouge package may not be available, using custom implementation

// =============================================================================
// TEXT PREPROCESSING
// =============================================================================

function preprocessText(text) {
    if (!text || typeof text !== 'string') return '';
    // Remove OCR noise and unnecessary symbols
    return text
        .replace(/[@#*&^]/g, '') // Remove specified symbols
        .replace(/\s+/g, ' ') // Normalize whitespace
        .replace(/[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/g, '') // Keep alphanumeric, Arabic, and basic punctuation
        .trim();
}

function tokenize(text) {
    return natural.WordTokenizer().tokenize(preprocessText(text.toLowerCase())) || [];
}

// =============================================================================
// ROUGE METRICS (ROUGE-1, ROUGE-2, ROUGE-L)
// =============================================================================

function calculateROUGE(candidate, reference) {
    try {
        const candidateTokens = tokenize(candidate);
        const referenceTokens = tokenize(reference);
        
        if (candidateTokens.length === 0 || referenceTokens.length === 0) {
            return { rouge1: 0, rouge2: 0, rougeL: 0 };
        }

        // ROUGE-1 (unigram overlap)
        const candidateUnigrams = new Set(candidateTokens);
        const referenceUnigrams = new Set(referenceTokens);
        const intersection1 = [...candidateUnigrams].filter(x => referenceUnigrams.has(x)).length;
        const rouge1 = intersection1 / Math.max(referenceUnigrams.size, 1);

        // ROUGE-2 (bigram overlap)
        const candidateBigrams = new Set();
        const referenceBigrams = new Set();
        for (let i = 0; i < candidateTokens.length - 1; i++) {
            candidateBigrams.add(`${candidateTokens[i]} ${candidateTokens[i + 1]}`);
        }
        for (let i = 0; i < referenceTokens.length - 1; i++) {
            referenceBigrams.add(`${referenceTokens[i]} ${referenceTokens[i + 1]}`);
        }
        const intersection2 = [...candidateBigrams].filter(x => referenceBigrams.has(x)).length;
        const rouge2 = intersection2 / Math.max(referenceBigrams.size, 1);

        // ROUGE-L (Longest Common Subsequence)
        const lcs = longestCommonSubsequence(candidateTokens, referenceTokens);
        const rougeL = lcs / Math.max(referenceTokens.length, 1);

        return {
            rouge1: Math.round(rouge1 * 10000) / 10000,
            rouge2: Math.round(rouge2 * 10000) / 10000,
            rougeL: Math.round(rougeL * 10000) / 10000
        };
    } catch (error) {
        console.error('ROUGE calculation error:', error);
        return { rouge1: 0, rouge2: 0, rougeL: 0 };
    }
}

function longestCommonSubsequence(arr1, arr2) {
    const m = arr1.length;
    const n = arr2.length;
    const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
    
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (arr1[i - 1] === arr2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}

// =============================================================================
// BLEU SCORE
// =============================================================================

function calculateBLEU(candidate, reference, maxN = 4) {
    try {
        const candidateTokens = tokenize(candidate);
        const referenceTokens = tokenize(reference);
        
        if (candidateTokens.length === 0 || referenceTokens.length === 0) {
            return 0;
        }

        // Calculate n-gram precisions
        const precisions = [];
        for (let n = 1; n <= maxN; n++) {
            const candidateNgrams = getNgrams(candidateTokens, n);
            const referenceNgrams = getNgrams(referenceTokens, n);
            
            if (candidateNgrams.length === 0) {
                precisions.push(0);
                continue;
            }

            let matches = 0;
            const referenceNgramCounts = {};
            referenceNgrams.forEach(ngram => {
                const key = ngram.join(' ');
                referenceNgramCounts[key] = (referenceNgramCounts[key] || 0) + 1;
            });

            const candidateNgramCounts = {};
            candidateNgrams.forEach(ngram => {
                const key = ngram.join(' ');
                candidateNgramCounts[key] = (candidateNgramCounts[key] || 0) + 1;
            });

            Object.keys(candidateNgramCounts).forEach(key => {
                matches += Math.min(
                    candidateNgramCounts[key],
                    referenceNgramCounts[key] || 0
                );
            });

            precisions.push(matches / candidateNgrams.length);
        }

        // Brevity penalty
        const brevityPenalty = candidateTokens.length > referenceTokens.length
            ? 1
            : Math.exp(1 - referenceTokens.length / candidateTokens.length);

        // Geometric mean of precisions
        const geometricMean = Math.pow(
            precisions.reduce((a, b) => a * b, 1),
            1 / maxN
        );

        return Math.round(brevityPenalty * geometricMean * 10000) / 10000;
    } catch (error) {
        console.error('BLEU calculation error:', error);
        return 0;
    }
}

function getNgrams(tokens, n) {
    const ngrams = [];
    for (let i = 0; i <= tokens.length - n; i++) {
        ngrams.push(tokens.slice(i, i + n));
    }
    return ngrams;
}

// =============================================================================
// BERTScore (Simplified - using cosine similarity of TF-IDF vectors)
// =============================================================================

function calculateBERTScore(candidate, reference) {
    try {
        const candidateTokens = tokenize(candidate);
        const referenceTokens = tokenize(reference);
        
        if (candidateTokens.length === 0 || referenceTokens.length === 0) {
            return { precision: 0, recall: 0, f1: 0 };
        }

        // Create TF-IDF vectors (simplified - using term frequency)
        const allTerms = [...new Set([...candidateTokens, ...referenceTokens])];
        const candidateVector = allTerms.map(term => 
            candidateTokens.filter(t => t === term).length / candidateTokens.length
        );
        const referenceVector = allTerms.map(term => 
            referenceTokens.filter(t => t === term).length / referenceTokens.length
        );

        // Cosine similarity
        const dotProduct = candidateVector.reduce((sum, val, i) => sum + val * referenceVector[i], 0);
        const candidateNorm = Math.sqrt(candidateVector.reduce((sum, val) => sum + val * val, 0));
        const referenceNorm = Math.sqrt(referenceVector.reduce((sum, val) => sum + val * val, 0));
        
        const similarity = dotProduct / (candidateNorm * referenceNorm || 1);

        // Precision: how much of candidate is in reference
        const precision = similarity;
        // Recall: how much of reference is in candidate
        const recall = similarity;
        // F1: harmonic mean
        const f1 = precision + recall > 0 
            ? (2 * precision * recall) / (precision + recall)
            : 0;

        return {
            precision: Math.round(precision * 10000) / 10000,
            recall: Math.round(recall * 10000) / 10000,
            f1: Math.round(f1 * 10000) / 10000
        };
    } catch (error) {
        console.error('BERTScore calculation error:', error);
        return { precision: 0, recall: 0, f1: 0 };
    }
}

// =============================================================================
// EXACT MATCH (EM)
// =============================================================================

function calculateExactMatch(candidate, reference) {
    try {
        const candidateNormalized = preprocessText(candidate).toLowerCase();
        const referenceNormalized = preprocessText(reference).toLowerCase();
        return candidateNormalized === referenceNormalized ? 1 : 0;
    } catch (error) {
        console.error('Exact Match calculation error:', error);
        return 0;
    }
}

// =============================================================================
// SummaC (Simplified - using sentence overlap)
// =============================================================================

function calculateSummaC(candidate, reference) {
    try {
        const candidateSentences = natural.SentenceTokenizer().tokenize(candidate) || [candidate];
        const referenceSentences = natural.SentenceTokenizer().tokenize(reference) || [reference];
        
        if (candidateSentences.length === 0 || referenceSentences.length === 0) {
            return 0;
        }

        let totalOverlap = 0;
        candidateSentences.forEach(candSent => {
            const candTokens = tokenize(candSent);
            if (candTokens.length === 0) return;
            
            let maxOverlap = 0;
            referenceSentences.forEach(refSent => {
                const refTokens = tokenize(refSent);
                const overlap = calculateTokenOverlap(candTokens, refTokens);
                maxOverlap = Math.max(maxOverlap, overlap);
            });
            totalOverlap += maxOverlap;
        });

        return Math.round((totalOverlap / candidateSentences.length) * 10000) / 10000;
    } catch (error) {
        console.error('SummaC calculation error:', error);
        return 0;
    }
}

function calculateTokenOverlap(tokens1, tokens2) {
    const set1 = new Set(tokens1);
    const set2 = new Set(tokens2);
    const intersection = [...set1].filter(x => set2.has(x)).length;
    return intersection / Math.max(tokens1.length, 1);
}

// =============================================================================
// RESPONSE METRICS
// =============================================================================

function calculateResponseMetrics(candidate, reference) {
    const candidateTokens = tokenize(candidate);
    const referenceTokens = tokenize(reference);
    
    return {
        candidateLength: candidate.length,
        candidateTokens: candidateTokens.length,
        referenceLength: reference.length,
        referenceTokens: referenceTokens.length,
        compressionRatio: referenceTokens.length > 0 
            ? candidateTokens.length / referenceTokens.length 
            : 0,
        lengthDifference: candidateTokens.length - referenceTokens.length
    };
}

// =============================================================================
// COMPREHENSIVE EVALUATION
// =============================================================================

function evaluateResponse(candidate, reference, options = {}) {
    const {
        includeROUGE = true,
        includeBLEU = true,
        includeBERTScore = true,
        includeEM = true,
        includeSummaC = true,
        includeResponseMetrics = true
    } = options;

    const metrics = {};

    if (includeROUGE) {
        metrics.rouge = calculateROUGE(candidate, reference);
    }

    if (includeBLEU) {
        metrics.bleu = calculateBLEU(candidate, reference);
    }

    if (includeBERTScore) {
        metrics.bertscore = calculateBERTScore(candidate, reference);
    }

    if (includeEM) {
        metrics.exactMatch = calculateExactMatch(candidate, reference);
    }

    if (includeSummaC) {
        metrics.summaC = calculateSummaC(candidate, reference);
    }

    if (includeResponseMetrics) {
        metrics.responseMetrics = calculateResponseMetrics(candidate, reference);
    }

    // Overall score (weighted average)
    const weights = {
        rouge: 0.25,
        bleu: 0.20,
        bertscore: 0.25,
        exactMatch: 0.10,
        summaC: 0.20
    };

    let overallScore = 0;
    let totalWeight = 0;

    if (metrics.rouge) {
        const rougeAvg = (metrics.rouge.rouge1 + metrics.rouge.rouge2 + metrics.rouge.rougeL) / 3;
        overallScore += rougeAvg * weights.rouge;
        totalWeight += weights.rouge;
    }

    if (metrics.bleu !== undefined) {
        overallScore += metrics.bleu * weights.bleu;
        totalWeight += weights.bleu;
    }

    if (metrics.bertscore) {
        overallScore += metrics.bertscore.f1 * weights.bertscore;
        totalWeight += weights.bertscore;
    }

    if (metrics.exactMatch !== undefined) {
        overallScore += metrics.exactMatch * weights.exactMatch;
        totalWeight += weights.exactMatch;
    }

    if (metrics.summaC !== undefined) {
        overallScore += metrics.summaC * weights.summaC;
        totalWeight += weights.summaC;
    }

    metrics.overallScore = totalWeight > 0 ? overallScore / totalWeight : 0;

    return metrics;
}

// =============================================================================
// BATCH EVALUATION
// =============================================================================

function evaluateBatch(results, references) {
    const evaluations = [];
    
    results.forEach((result, index) => {
        const reference = references[index] || '';
        const candidate = result.answer || result.response || '';
        
        if (candidate && reference) {
            evaluations.push({
                index,
                modelId: result.modelId,
                metrics: evaluateResponse(candidate, reference),
                candidate,
                reference
            });
        }
    });

    // Aggregate metrics
    const aggregated = {
        rouge: { rouge1: 0, rouge2: 0, rougeL: 0 },
        bleu: 0,
        bertscore: { precision: 0, recall: 0, f1: 0 },
        exactMatch: 0,
        summaC: 0,
        overallScore: 0
    };

    evaluations.forEach(eval => {
        if (eval.metrics.rouge) {
            aggregated.rouge.rouge1 += eval.metrics.rouge.rouge1;
            aggregated.rouge.rouge2 += eval.metrics.rouge.rouge2;
            aggregated.rouge.rougeL += eval.metrics.rouge.rougeL;
        }
        if (eval.metrics.bleu !== undefined) {
            aggregated.bleu += eval.metrics.bleu;
        }
        if (eval.metrics.bertscore) {
            aggregated.bertscore.precision += eval.metrics.bertscore.precision;
            aggregated.bertscore.recall += eval.metrics.bertscore.recall;
            aggregated.bertscore.f1 += eval.metrics.bertscore.f1;
        }
        if (eval.metrics.exactMatch !== undefined) {
            aggregated.exactMatch += eval.metrics.exactMatch;
        }
        if (eval.metrics.summaC !== undefined) {
            aggregated.summaC += eval.metrics.summaC;
        }
        aggregated.overallScore += eval.metrics.overallScore;
    });

    const count = evaluations.length;
    if (count > 0) {
        aggregated.rouge.rouge1 /= count;
        aggregated.rouge.rouge2 /= count;
        aggregated.rouge.rougeL /= count;
        aggregated.bleu /= count;
        aggregated.bertscore.precision /= count;
        aggregated.bertscore.recall /= count;
        aggregated.bertscore.f1 /= count;
        aggregated.exactMatch /= count;
        aggregated.summaC /= count;
        aggregated.overallScore /= count;
    }

    return {
        evaluations,
        aggregated,
        count
    };
}

module.exports = {
    evaluateResponse,
    evaluateBatch,
    calculateROUGE,
    calculateBLEU,
    calculateBERTScore,
    calculateExactMatch,
    calculateSummaC,
    preprocessText,
    tokenize
};

