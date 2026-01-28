const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const multer = require('multer');
const Tesseract = require('tesseract.js');
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
// Optional dependencies for enhanced features
let sharp = null;
try {
    sharp = require('sharp');
} catch (err) {
    console.warn('Warning: sharp not available. Image preprocessing will be skipped.');
}

let pdfjsLib = null;
let createCanvas = null;
try {
    pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');
    const canvas = require('canvas');
    createCanvas = canvas.createCanvas;
} catch (err) {
    console.warn('Warning: PDF support libraries not fully available. PDF conversion may fail.');
}
require('dotenv').config();

// Import evaluation module
const evaluation = require('./evaluation');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// =============================================================================
// LOGGING UTILITY
// =============================================================================

const logger = {
    info: (message, data = {}) => {
        console.log(`[INFO] ${new Date().toISOString()} - ${message}`, Object.keys(data).length > 0 ? JSON.stringify(data, null, 2) : '');
    },
    error: (message, error = {}) => {
        console.error(`[ERROR] ${new Date().toISOString()} - ${message}`, error.message || error);
        if (error.stack) console.error(error.stack);
    },
    warn: (message, data = {}) => {
        console.warn(`[WARN] ${new Date().toISOString()} - ${message}`, data);
    },
    debug: (message, data = {}) => {
        if (process.env.DEBUG === 'true') {
            console.log(`[DEBUG] ${new Date().toISOString()} - ${message}`, data);
        }
    }
};

// =============================================================================
// FILE UPLOAD CONFIGURATION
// =============================================================================

const upload = multer({
    dest: 'uploads/',
    limits: { fileSize: 20 * 1024 * 1024 }, // 20MB limit for PDFs
    fileFilter: (req, file, cb) => {
        const allowedTypes = [
            'image/jpeg',
            'image/png',
            'image/gif',
            'image/bmp',
            'image/webp',
            'image/tiff',
            'application/pdf'
        ];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Supported: JPG, PNG, GIF, BMP, WebP, TIFF, PDF'), false);
        }
    }
});

// Ensure directories exist
const uploadsDir = path.join(__dirname, 'uploads');
const tempDir = path.join(__dirname, 'temp');
fs.mkdir(uploadsDir, { recursive: true }).catch(console.error);
fs.mkdir(tempDir, { recursive: true }).catch(console.error);

// =============================================================================
// API KEYS & PROVIDER INITIALIZATION
// =============================================================================

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Initialize Gemini
const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

// =============================================================================
// MODEL CONFIGURATIONS
// =============================================================================

const MODEL_CONFIG = {
    llm: {
        'mistral-7b-instruct': {
            provider: 'openrouter',
            modelId: 'mistralai/mistral-7b-instruct:free',
            name: 'Mistral 7B Instruct',
            icon: '🌐',
            cost: 'free',
            speed: 'fast',
            quality: 4
        },
        'llama-3.1-8b-instant': {
            provider: 'groq',
            modelId: 'llama-3.1-8b-instant',
            name: 'Llama 3.1 8B Instant',
            icon: '⚡',
            cost: 'free',
            speed: 'fastest',
            quality: 4
        },
        'llama-3.3-70b-versatile': {
            provider: 'groq',
            modelId: 'llama-3.3-70b-versatile',
            name: 'Llama 3.3 70B Versatile',
            icon: '⚡',
            cost: 'free',
            speed: 'very-fast',
            quality: 5
        }
    },
    ocr: {
        'tesseract-default': {
            name: 'Tesseract Default',
            icon: '📄',
            lang: 'eng',
            oem: 3, // Default - uses whatever is available
            psm: 3, // Fully automatic page segmentation
            description: 'Standard OCR with automatic page segmentation',
            bestFor: 'General purpose text extraction'
        },
        'tesseract-accurate': {
            name: 'Tesseract LSTM (Best)',
            icon: '🎯',
            lang: 'eng',
            oem: 1, // LSTM only
            psm: 3,
            description: 'LSTM neural network for maximum accuracy',
            bestFor: 'Best accuracy for clear printed text'
        },
        'tesseract-legacy': {
            name: 'Tesseract Legacy',
            icon: '📜',
            lang: 'eng',
            oem: 0, // Legacy engine only
            psm: 3,
            description: 'Legacy Tesseract engine',
            bestFor: 'Older documents, typewritten text'
        },
        'tesseract-auto-osd': {
            name: 'Tesseract Auto + OSD',
            icon: '🔄',
            lang: 'eng',
            oem: 3,
            psm: 1, // Automatic with OSD
            description: 'Auto page segmentation with orientation detection',
            bestFor: 'Rotated or skewed documents'
        },
        'tesseract-single-column': {
            name: 'Tesseract Single Column',
            icon: '📃',
            lang: 'eng',
            oem: 3,
            psm: 4, // Single column of text
            description: 'Assumes single column of variable text',
            bestFor: 'Single column articles, letters'
        },
        'tesseract-block': {
            name: 'Tesseract Text Block',
            icon: '⬜',
            lang: 'eng',
            oem: 3,
            psm: 6, // Uniform block
            description: 'Assumes uniform block of text',
            bestFor: 'Paragraphs, uniform text blocks'
        },
        'tesseract-single-line': {
            name: 'Tesseract Single Line',
            icon: '➖',
            lang: 'eng',
            oem: 3,
            psm: 7, // Single text line
            description: 'Treats image as single text line',
            bestFor: 'Headers, titles, single lines'
        },
        'tesseract-single-word': {
            name: 'Tesseract Single Word',
            icon: '🔤',
            lang: 'eng',
            oem: 3,
            psm: 8, // Single word
            description: 'Treats image as single word',
            bestFor: 'Single words, labels'
        },
        'tesseract-sparse': {
            name: 'Tesseract Sparse Text',
            icon: '🔍',
            lang: 'eng',
            oem: 3,
            psm: 11, // Sparse text
            description: 'Find text without order assumption',
            bestFor: 'Scattered text, diagrams with labels'
        },
        'tesseract-raw': {
            name: 'Tesseract Raw Line',
            icon: '📏',
            lang: 'eng',
            oem: 3,
            psm: 13, // Raw line
            description: 'Raw line without preprocessing',
            bestFor: 'Clean single lines without noise'
        },
        'tesseract-arabic': {
            name: 'Tesseract Arabic',
            icon: '🕌',
            lang: 'ara',
            oem: 3,
            psm: 3,
            description: 'Optimized for Arabic text recognition',
            bestFor: 'Arabic documents and text'
        },
        'tesseract-multilang': {
            name: 'Tesseract Multi-Language',
            icon: '🌐',
            lang: 'eng+ara',
            oem: 3,
            psm: 3,
            description: 'Supports English and Arabic simultaneously',
            bestFor: 'Mixed English/Arabic documents'
        },
        'tesseract-arabic-accurate': {
            name: 'Tesseract Arabic LSTM',
            icon: '🎯',
            lang: 'ara',
            oem: 1,
            psm: 3,
            description: 'LSTM neural network for Arabic text',
            bestFor: 'High accuracy Arabic OCR'
        }
    }
};

// =============================================================================
// STATISTICS TRACKING
// =============================================================================

const stats = {
    totalRequests: 0,
    llmRequests: 0,
    ocrRequests: 0,
    comparisons: 0,
    byModel: {},
    errors: 0,
    startTime: Date.now()
};

// =============================================================================
// PROMPT BUILDERS
// =============================================================================

function buildPrompt(taskType, subject, question) {
    const subjectContext = subject ? `in the context of ${subject}` : '';

    const prompts = {
        qa: `You are an expert educational assistant. Answer the following question ${subjectContext} accurately and comprehensively.

Question: ${question}

Instructions:
- Provide a clear, well-structured answer
- Use examples where helpful
- Explain any technical terms
- Make it appropriate for a student learning this topic
- Be thorough but concise`,

        summarize: `You are an expert at creating clear, concise summaries. Summarize the following content ${subjectContext}.

Content to summarize:
${question}

Instructions:
- Capture the main ideas and key points
- Organize logically with clear structure
- Use bullet points for clarity when appropriate
- Preserve important details while being concise
- Highlight any critical information`,

        explain: `You are a patient and skilled teacher. Explain the following concept ${subjectContext} in a way that's easy to understand.

Concept to explain: ${question}

Instructions:
- Start with a simple overview
- Break down complex ideas into smaller parts
- Use analogies and real-world examples
- Build understanding step by step
- Anticipate and address common misconceptions
- Include practical applications where relevant`
    };

    return prompts[taskType] || prompts.qa;
}

// =============================================================================
// TEXT FILTERING
// =============================================================================

function filterResponseText(text) {
    if (!text || typeof text !== 'string') {
        return text;
    }
    // Remove specific symbols: @ # * & ^
    return text.replace(/[@#*&^]/g, '');
}

// =============================================================================
// LLM PROVIDER FUNCTIONS
// =============================================================================

async function callOpenRouter(prompt, modelId, options = {}) {
    if (!OPENROUTER_API_KEY) {
        throw new Error('OpenRouter API key not configured');
    }

    try {
        const requestBody = {
            model: modelId,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: options.max_tokens || 4000,
            temperature: options.temperature || 0.7
        };

        logger.debug('OpenRouter request', { modelId, promptLength: prompt.length, maxTokens: requestBody.max_tokens });

        const response = await axios.post(
            'https://openrouter.ai/api/v1/chat/completions',
            requestBody,
            {
                headers: {
                    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'http://localhost:3000',
                    'X-Title': 'Educational AI Testing Platform'
                },
                timeout: 120000
            }
        );

        if (!response.data || !response.data.choices || !response.data.choices[0]) {
            throw new Error('Invalid response format from OpenRouter');
        }

        const content = response.data.choices[0].message?.content;
        if (!content) {
            throw new Error('Empty response from OpenRouter');
        }

        return filterResponseText(content);
    } catch (error) {
        if (error.response) {
            const errorMsg = error.response.data?.error?.message || error.response.statusText;
            logger.error('OpenRouter API error', { modelId, status: error.response.status, error: errorMsg });
            throw new Error(`OpenRouter Error: ${errorMsg}`);
        }
        logger.error('OpenRouter request error', error);
        throw error;
    }
}

async function callGroq(prompt, modelId, options = {}) {
    if (!GROQ_API_KEY) {
        throw new Error('Groq API key not configured');
    }

    try {
        // Validate model ID format
        const validGroqModels = [
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            'llama-3.1-70b-versatile',
            'gemma-7b-it',
            'mixtral-8x22b-instruct'
        ];

        if (!validGroqModels.includes(modelId)) {
            logger.warn('Potentially invalid Groq model ID', { modelId });
        }

        const requestBody = {
            model: modelId,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: options.max_tokens || 4000,
            temperature: options.temperature || 0.7
        };

        logger.debug('Groq request', { modelId, promptLength: prompt.length, maxTokens: requestBody.max_tokens });

        const response = await axios.post(
            'https://api.groq.com/openai/v1/chat/completions',
            requestBody,
            {
                headers: {
                    'Authorization': `Bearer ${GROQ_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 120000
            }
        );

        if (!response.data || !response.data.choices || !response.data.choices[0]) {
            throw new Error('Invalid response format from Groq');
        }

        const content = response.data.choices[0].message?.content;
        if (!content) {
            throw new Error('Empty response from Groq');
        }

        return filterResponseText(content);
    } catch (error) {
        if (error.response) {
            const errorMsg = error.response.data?.error?.message || error.response.statusText;
            logger.error('Groq API error', { modelId, status: error.response.status, error: errorMsg });
            
            // Handle specific Groq errors
            if (errorMsg.includes('decommissioned')) {
                throw new Error(`Groq Error: Model ${modelId} has been decommissioned. Please use a different model.`);
            }
            if (errorMsg.includes('rate limit') || error.response.status === 429) {
                throw new Error('Groq Error: Rate limit exceeded. Please wait and try again.');
            }
            
            throw new Error(`Groq Error: ${errorMsg}`);
        }
        logger.error('Groq request error', error);
        throw error;
    }
}

async function callGemini(prompt, modelId, options = {}) {
    if (!genAI) {
        throw new Error('Gemini API key not configured');
    }

    try {
        // Validate and normalize model ID
        const modelIdMap = {
            'gemini-2.0-flash-exp': 'gemini-2.0-flash-exp',
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-pro': 'gemini-1.5-pro' // Map old name to current
        };

        const normalizedModelId = modelIdMap[modelId] || modelId;
        
        logger.debug('Gemini request', { modelId, normalizedModelId, promptLength: prompt.length });

        const generationConfig = {
            maxOutputTokens: options.max_tokens || 4000,
            temperature: options.temperature || 0.7
        };

        const model = genAI.getGenerativeModel({ 
            model: normalizedModelId,
            generationConfig
        });
        
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();
        
        if (!text || text.trim().length === 0) {
            throw new Error('Empty response from Gemini');
        }

        return filterResponseText(text);
    } catch (error) {
        const errorMessage = error.message || 'Unknown Gemini error';
        logger.error('Gemini API error', { modelId, error: errorMessage });
        
        if (errorMessage.includes('429') || errorMessage.includes('quota') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
            throw new Error('Gemini quota exceeded. Please wait and try again.');
        }
        if (errorMessage.includes('404') || errorMessage.includes('not found') || errorMessage.includes('NOT_FOUND')) {
            throw new Error(`Gemini model "${modelId}" not available.`);
        }
        if (errorMessage.includes('PERMISSION_DENIED') || errorMessage.includes('API_KEY')) {
            throw new Error('Gemini API key invalid or permission denied.');
        }
        
        throw new Error(`Gemini Error: ${errorMessage}`);
    }
}

async function callLLM(modelKey, prompt, options = {}) {
    const config = MODEL_CONFIG.llm[modelKey];
    if (!config) {
        throw new Error(`Unknown model: ${modelKey}`);
    }

    const startTime = Date.now();
    logger.debug(`Calling LLM`, { modelKey, provider: config.provider, promptLength: prompt.length });

    try {
        let result;
        switch (config.provider) {
            case 'openrouter':
                result = await callOpenRouter(prompt, config.modelId, options);
                break;
            case 'groq':
                result = await callGroq(prompt, config.modelId, options);
                break;
            case 'google':
                result = await callGemini(prompt, config.modelId, options);
                break;
            default:
                throw new Error(`Unknown provider: ${config.provider}`);
        }

        const responseTime = Date.now() - startTime;
        logger.debug(`LLM call completed`, { modelKey, responseTime, resultLength: result?.length || 0 });
        
        return result;
    } catch (error) {
        const responseTime = Date.now() - startTime;
        logger.error(`LLM call failed`, { modelKey, provider: config.provider, responseTime, error });
        throw error;
    }
}

// =============================================================================
// IMAGE PREPROCESSING
// =============================================================================

async function preprocessImage(inputPath, outputPath) {
    try {
        // Check if sharp is available
        if (!sharp) {
            console.warn('  Sharp not available, using original image without preprocessing');
            await fs.copyFile(inputPath, outputPath);
            return outputPath;
        }
        
        // Read the image and preprocess for better OCR
        await sharp(inputPath)
            .greyscale() // Convert to grayscale
            .normalise() // Normalize contrast
            .sharpen() // Sharpen edges
            .png() // Convert to PNG
            .toFile(outputPath);
        
        console.log(`  Image preprocessed: ${outputPath}`);
        return outputPath;
    } catch (error) {
        console.warn(`  Image preprocessing failed (${error.message}), using original image`);
        // If preprocessing fails, copy the original
        try {
            await fs.copyFile(inputPath, outputPath);
            return outputPath;
        } catch (copyError) {
            // If copy fails, just return original path
            console.warn(`  Could not copy file, using original: ${copyError.message}`);
            return inputPath;
        }
    }
}

// =============================================================================
// PDF TO IMAGE CONVERSION
// =============================================================================

async function convertPdfToImages(pdfPath, options = {}) {
    const images = [];
    const maxPages = options.maxPages || 50; // Limit to prevent memory issues
    const chunkSize = options.chunkSize || 10; // Process in chunks
    
    logger.info('Converting PDF to images', { pdfPath, maxPages, chunkSize });
    
    try {
        // Try using pdf-poppler if available
        try {
            const pdfPoppler = require('pdf-poppler');
            
            const opts = {
                format: 'png',
                out_dir: tempDir,
                out_prefix: `pdf_${Date.now()}_${path.basename(pdfPath, '.pdf')}`,
                scale: 2048 // Good resolution for OCR
            };
            
            await pdfPoppler.convert(pdfPath, opts);
            
            // Find generated images
            const files = await fs.readdir(tempDir);
            const pdfImages = files
                .filter(f => f.startsWith(opts.out_prefix) && f.endsWith('.png'))
                .sort()
                .slice(0, maxPages) // Limit pages
                .map(f => path.join(tempDir, f));
            
            images.push(...pdfImages);
            logger.info('PDF converted using pdf-poppler', { imageCount: images.length });
            
        } catch (popplerError) {
            logger.debug('pdf-poppler not available, trying alternative method...', { error: popplerError.message });
            
            // Alternative: Use pdfjs-dist with canvas
            if (!pdfjsLib || !createCanvas) {
                throw new Error('PDF conversion libraries not available');
            }
            
            const data = new Uint8Array(await fs.readFile(pdfPath));
            const pdf = await pdfjsLib.getDocument({ data }).promise;
            
            const totalPages = Math.min(pdf.numPages, maxPages);
            logger.info('Processing PDF pages', { totalPages, maxPages });
            
            // Process in chunks to manage memory
            for (let chunkStart = 1; chunkStart <= totalPages; chunkStart += chunkSize) {
                const chunkEnd = Math.min(chunkStart + chunkSize - 1, totalPages);
                logger.debug(`Processing PDF chunk`, { chunkStart, chunkEnd });
                
                for (let i = chunkStart; i <= chunkEnd; i++) {
                    try {
                        const page = await pdf.getPage(i);
                        const scale = 2.0; // Good for OCR
                        const viewport = page.getViewport({ scale });
                        
                        const canvas = createCanvas(viewport.width, viewport.height);
                        const context = canvas.getContext('2d');
                        
                        await page.render({
                            canvasContext: context,
                            viewport: viewport
                        }).promise;
                        
                        const imagePath = path.join(tempDir, `page_${Date.now()}_${i}.png`);
                        const buffer = canvas.toBuffer('image/png');
                        await fs.writeFile(imagePath, buffer);
                        images.push(imagePath);
                        
                        // Clear canvas from memory
                        canvas.width = 0;
                        canvas.height = 0;
                    } catch (pageError) {
                        logger.error(`Error processing PDF page ${i}`, { error: pageError.message });
                        // Continue with next page
                    }
                }
                
                // Small delay between chunks to allow garbage collection
                if (chunkEnd < totalPages) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }
            
            logger.info('PDF converted using pdfjs-dist', { imageCount: images.length });
        }
        
    } catch (error) {
        logger.error('PDF conversion failed', error);
        // If both methods fail, provide helpful error message
        if (error.message.includes('Cannot find module') || error.message.includes('require')) {
            throw new Error('PDF support requires additional packages. Run: npm install pdf-poppler pdfjs-dist canvas');
        }
        throw new Error(`PDF conversion failed: ${error.message}`);
    }
    
    if (images.length === 0) {
        throw new Error('No images generated from PDF');
    }
    
    return images;
}

// =============================================================================
// OCR PROCESSING FUNCTIONS - COMPLETELY FIXED
// =============================================================================

async function processOCR(filePath, modelKey, originalName = '') {
    const config = MODEL_CONFIG.ocr[modelKey];
    if (!config) {
        throw new Error(`Unknown OCR model: ${modelKey}`);
    }

    logger.info('Processing OCR', { modelKey, fileName: originalName });

    // Check if file is PDF
    let isPdf = false;
    try {
        isPdf = originalName && originalName.toLowerCase().endsWith('.pdf');
        if (!isPdf) {
            // Check file signature (first 4 bytes: %PDF)
            const buffer = await fs.readFile(filePath);
            const header = buffer.slice(0, 4).toString('hex');
            isPdf = header === '25504446'; // %PDF in hex
        }
    } catch (err) {
        logger.warn('Error checking PDF', { error: err.message });
    }

    let imagesToProcess = [];
    let tempFiles = [];

    try {
        if (isPdf) {
            logger.info(`Converting PDF to images for OCR...`);
            imagesToProcess = await convertPdfToImages(filePath, { maxPages: 50, chunkSize: 10 });
            tempFiles = [...imagesToProcess];
            logger.info(`Converted PDF to ${imagesToProcess.length} images`);
        } else {
            // Preprocess single image
            const preprocessedPath = path.join(tempDir, `preprocessed_${Date.now()}.png`);
            await preprocessImage(filePath, preprocessedPath);
            imagesToProcess = [preprocessedPath];
            tempFiles = [preprocessedPath];
        }

        if (imagesToProcess.length === 0) {
            throw new Error('No images to process');
        }

        let allText = '';
        let totalConfidence = 0;
        let totalWords = 0;
        let totalLines = 0;
        let processedPages = 0;

        // Process images in chunks to manage memory
        const ocrChunkSize = 5;
        for (let chunkStart = 0; chunkStart < imagesToProcess.length; chunkStart += ocrChunkSize) {
            const chunkEnd = Math.min(chunkStart + ocrChunkSize, imagesToProcess.length);
            logger.debug(`Processing OCR chunk`, { chunkStart: chunkStart + 1, chunkEnd, total: imagesToProcess.length });

            for (let i = chunkStart; i < chunkEnd; i++) {
                const imagePath = imagesToProcess[i];
                logger.debug(`Processing image ${i + 1}/${imagesToProcess.length} with ${modelKey}...`);

                let worker = null;
                try {
                    // Create worker with specific OEM
                    // OEM 0 = Legacy, 1 = LSTM, 2 = Legacy+LSTM, 3 = Default
                    worker = await Tesseract.createWorker(config.lang, config.oem, {
                        logger: m => {
                            if (m.status === 'recognizing text' && m.progress) {
                                logger.debug('OCR progress', { progress: Math.round(m.progress * 100) });
                            }
                        },
                        errorHandler: err => logger.error('Tesseract error', { error: err.message })
                    });

                    // Set PSM (Page Segmentation Mode)
                    await worker.setParameters({
                        tessedit_pageseg_mode: config.psm.toString(),
                    });

                    // Recognize
                    logger.debug(`Recognizing text from image`, { imagePath, page: i + 1 });
                    const { data } = await worker.recognize(imagePath);

                    const extractedText = data.text || '';
                    const hasText = extractedText.trim().length > 0;
                    
                    logger.debug(`OCR extraction complete`, { 
                        page: i + 1,
                        characters: extractedText.length,
                        confidence: (data.confidence || 0).toFixed(1)
                    });

                    if (hasText) {
                        // Clean OCR text using evaluation preprocessing
                        const cleanedText = evaluation.preprocessText(extractedText);
                        
                        if (imagesToProcess.length > 1) {
                            allText += `\n--- Page ${i + 1} ---\n`;
                        }
                        allText += cleanedText;
                        totalConfidence += data.confidence || 0;
                        totalWords += data.words?.length || 0;
                        totalLines += data.lines?.length || 0;
                        processedPages++;
                    } else {
                        logger.warn(`No text detected in page ${i + 1}`);
                    }

                    await worker.terminate();
                    worker = null;

                } catch (pageError) {
                    logger.error(`Error processing page ${i + 1}`, { error: pageError.message });
                    if (worker) {
                        await worker.terminate().catch(() => {});
                    }
                }
            }

            // Small delay between chunks for memory management
            if (chunkEnd < imagesToProcess.length) {
                await new Promise(resolve => setTimeout(resolve, 200));
            }
        }

        // Cleanup temp files
        for (const tempFile of tempFiles) {
            await fs.unlink(tempFile).catch(() => {});
        }

        if (!allText.trim()) {
            return {
                text: '[No text detected in the image. Try a different OCR mode or ensure the image contains clear text.]',
                confidence: 0,
                words: 0,
                lines: 0,
                pages: imagesToProcess.length
            };
        }

        return {
            text: allText.trim(),
            confidence: processedPages > 0 ? totalConfidence / processedPages : 0,
            words: totalWords,
            lines: totalLines,
            pages: imagesToProcess.length
        };

    } catch (error) {
        // Cleanup on error
        for (const tempFile of tempFiles) {
            await fs.unlink(tempFile).catch(() => {});
        }
        throw error;
    }
}

async function processOCRComparison(filePath, modelKeys, originalName = '') {
    const results = [];

    for (const modelKey of modelKeys) {
        const startTime = Date.now();
        try {
            console.log(`\n--- Processing with ${modelKey} ---`);
            const result = await processOCR(filePath, modelKey, originalName);
            const responseTime = Date.now() - startTime;

            stats.byModel[modelKey] = (stats.byModel[modelKey] || 0) + 1;

            results.push({
                modelId: modelKey,
                model: MODEL_CONFIG.ocr[modelKey],
                text: result.text,
                confidence: result.confidence,
                words: result.words,
                lines: result.lines,
                pages: result.pages,
                responseTime,
                success: true
            });
        } catch (error) {
            console.error(`Error with ${modelKey}:`, error.message);
            results.push({
                modelId: modelKey,
                model: MODEL_CONFIG.ocr[modelKey],
                error: error.message,
                responseTime: Date.now() - startTime,
                success: false
            });
        }
    }

    return results;
}

// =============================================================================
// API ROUTES - LLM
// =============================================================================

app.post('/api/ask', async (req, res) => {
    stats.totalRequests++;
    stats.llmRequests++;

    try {
        const { question, subject, taskType, modelId } = req.body;

        if (!question?.trim()) {
            return res.status(400).json({ success: false, error: 'Question is required' });
        }

        if (!modelId) {
            return res.status(400).json({ success: false, error: 'Model ID is required' });
        }

        if (!MODEL_CONFIG.llm[modelId]) {
            return res.status(400).json({ success: false, error: `Invalid model ID: ${modelId}` });
        }

        console.log(`Processing LLM request - Model: ${modelId}, Task: ${taskType || 'qa'}`);

        const prompt = buildPrompt(taskType || 'qa', subject, question);
        const startTime = Date.now();

        const answer = await callLLM(modelId, prompt);
        const responseTime = Date.now() - startTime;

        stats.byModel[modelId] = (stats.byModel[modelId] || 0) + 1;

        console.log(`LLM request completed in ${responseTime}ms`);

        res.json({
            success: true,
            answer,
            responseTime,
            model: MODEL_CONFIG.llm[modelId],
            taskType: taskType || 'qa',
            subject: subject || 'General'
        });

    } catch (error) {
        stats.errors++;
        console.error('LLM Error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/compare', async (req, res) => {
    stats.totalRequests++;
    stats.comparisons++;

    try {
        const { question, subject, taskType, modelIds } = req.body;

        if (!question?.trim()) {
            return res.status(400).json({ success: false, error: 'Question is required' });
        }

        if (!modelIds || !Array.isArray(modelIds) || modelIds.length < 2) {
            return res.status(400).json({ success: false, error: 'At least 2 model IDs required' });
        }

        if (modelIds.length > 5) {
            return res.status(400).json({ success: false, error: 'Maximum 5 models allowed' });
        }

        for (const modelId of modelIds) {
            if (!MODEL_CONFIG.llm[modelId]) {
                return res.status(400).json({ success: false, error: `Invalid model ID: ${modelId}` });
            }
        }

        console.log(`Processing LLM comparison - Models: ${modelIds.join(', ')}`);

        const prompt = buildPrompt(taskType || 'qa', subject, question);

        const results = await Promise.all(
            modelIds.map(async (modelId) => {
                const startTime = Date.now();
                try {
                    const answer = await callLLM(modelId, prompt);
                    const responseTime = Date.now() - startTime;
                    stats.byModel[modelId] = (stats.byModel[modelId] || 0) + 1;
                    return { modelId, model: MODEL_CONFIG.llm[modelId], answer, responseTime, success: true };
                } catch (error) {
                    return { modelId, model: MODEL_CONFIG.llm[modelId], error: error.message, responseTime: Date.now() - startTime, success: false };
                }
            })
        );

        res.json({ success: true, results, query: { question, subject, taskType: taskType || 'qa' } });

    } catch (error) {
        stats.errors++;
        console.error('Compare Error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// =============================================================================
// API ROUTES - OCR
// =============================================================================

app.post('/api/ocr', upload.single('document'), async (req, res) => {
    stats.totalRequests++;
    stats.ocrRequests++;

    try {
        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No file uploaded' });
        }

        const modelId = req.body.modelId;
        if (!modelId) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: 'Model ID is required' });
        }

        if (!MODEL_CONFIG.ocr[modelId]) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: `Invalid OCR model: ${modelId}` });
        }

        console.log(`\nProcessing OCR - Model: ${modelId}, File: ${req.file.originalname}`);

        const startTime = Date.now();
        const result = await processOCR(req.file.path, modelId, req.file.originalname);
        const responseTime = Date.now() - startTime;

        await fs.unlink(req.file.path).catch(() => {});

        stats.byModel[modelId] = (stats.byModel[modelId] || 0) + 1;

        console.log(`OCR completed in ${responseTime}ms - ${result.words} words, ${result.confidence.toFixed(1)}% confidence`);

        res.json({
            success: true,
            text: result.text,
            confidence: result.confidence,
            words: result.words,
            lines: result.lines,
            pages: result.pages,
            responseTime,
            model: MODEL_CONFIG.ocr[modelId],
            fileName: req.file.originalname
        });

    } catch (error) {
        stats.errors++;
        if (req.file) await fs.unlink(req.file.path).catch(() => {});
        console.error('OCR Error:', error);
        console.error('Error stack:', error.stack);
        
        // Provide more helpful error messages
        let errorMessage = error.message;
        if (error.message.includes('sharp') || error.message.includes('preprocessing')) {
            errorMessage = 'Image processing failed. Ensure sharp package is installed: npm install sharp';
        } else if (error.message.includes('Tesseract') || error.message.includes('worker')) {
            errorMessage = 'OCR engine error: ' + error.message + '. Check that eng.traineddata is available.';
        } else if (error.message.includes('PDF')) {
            errorMessage = error.message;
        }
        
        res.status(500).json({ success: false, error: errorMessage });
    }
});

app.post('/api/ocr-compare', upload.single('document'), async (req, res) => {
    stats.totalRequests++;
    stats.comparisons++;

    try {
        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No file uploaded' });
        }

        let modelIds;
        try {
            modelIds = JSON.parse(req.body.modelIds);
        } catch {
            modelIds = req.body.modelIds;
        }

        if (!modelIds || !Array.isArray(modelIds) || modelIds.length < 2) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: 'At least 2 model IDs required' });
        }

        if (modelIds.length > 5) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: 'Maximum 5 models allowed' });
        }

        for (const modelId of modelIds) {
            if (!MODEL_CONFIG.ocr[modelId]) {
                await fs.unlink(req.file.path).catch(() => {});
                return res.status(400).json({ success: false, error: `Invalid OCR model: ${modelId}` });
            }
        }

        console.log(`\nProcessing OCR comparison - Models: ${modelIds.join(', ')}, File: ${req.file.originalname}`);

        const results = await processOCRComparison(req.file.path, modelIds, req.file.originalname);

        await fs.unlink(req.file.path).catch(() => {});

        console.log(`OCR comparison completed`);

        res.json({ success: true, results, fileName: req.file.originalname });

    } catch (error) {
        stats.errors++;
        if (req.file) await fs.unlink(req.file.path).catch(() => {});
        console.error('OCR Compare Error:', error);
        console.error('Error stack:', error.stack);
        
        // Provide more helpful error messages
        let errorMessage = error.message;
        if (error.message.includes('sharp') || error.message.includes('preprocessing')) {
            errorMessage = 'Image processing failed. Ensure sharp package is installed: npm install sharp';
        } else if (error.message.includes('Tesseract') || error.message.includes('worker')) {
            errorMessage = 'OCR engine error: ' + error.message + '. Check that eng.traineddata is available.';
        } else if (error.message.includes('PDF')) {
            errorMessage = error.message;
        }
        
        res.status(500).json({ success: false, error: errorMessage });
    }
});

// =============================================================================
// API ROUTES - UTILITY
// =============================================================================

app.get('/api/models', (req, res) => {
    res.json({
        success: true,
        models: {
            llm: Object.entries(MODEL_CONFIG.llm).map(([id, config]) => ({ id, ...config })),
            ocr: Object.entries(MODEL_CONFIG.ocr).map(([id, config]) => ({ id, ...config }))
        }
    });
});

app.get('/api/stats', (req, res) => {
    res.json({
        success: true,
        stats: {
            ...stats,
            uptime: Math.floor((Date.now() - stats.startTime) / 1000),
            memoryUsage: process.memoryUsage()
        }
    });
});

app.get('/api/health', (req, res) => {
    res.json({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        providers: {
            openrouter: !!OPENROUTER_API_KEY,
            groq: !!GROQ_API_KEY,
            gemini: !!GEMINI_API_KEY
        }
    });
});

// =============================================================================
// API ROUTES - BATCH EVALUATION
// =============================================================================

app.post('/api/batch-eval', async (req, res) => {
    stats.totalRequests++;
    stats.comparisons++;

    const evaluationStartTime = Date.now();
    logger.info('Batch evaluation started', { 
        queryCount: req.body.queries?.length || 0,
        modelCount: req.body.modelIds?.length || 0,
        runs: req.body.runs || 1
    });

    try {
        const { queries, modelIds, runs = 1 } = req.body;

        if (!queries || !Array.isArray(queries) || queries.length === 0) {
            return res.status(400).json({ success: false, error: 'Queries array is required' });
        }

        if (!modelIds || !Array.isArray(modelIds) || modelIds.length < 2) {
            return res.status(400).json({ success: false, error: 'At least 2 model IDs required' });
        }

        if (modelIds.length > 5) {
            return res.status(400).json({ success: false, error: 'Maximum 5 models allowed' });
        }

        // Validate all model IDs
        for (const modelId of modelIds) {
            if (!MODEL_CONFIG.llm[modelId]) {
                logger.error('Invalid model ID in batch evaluation', { modelId });
                return res.status(400).json({ success: false, error: `Invalid model ID: ${modelId}` });
            }
        }

        logger.info(`Processing batch evaluation - ${queries.length} queries, ${modelIds.length} models, ${runs} runs`);

        const results = [];
        const evaluationMetrics = [];

        // Process queries with rate limiting
        for (let i = 0; i < queries.length; i++) {
            const query = queries[i];
            logger.debug(`Processing query ${i + 1}/${queries.length}`, { 
                queryId: query.id || i,
                task: query.taskType || query.task,
                subject: query.subject 
            });
            
            // Run multiple times if requested
            for (let run = 0; run < runs; run++) {
                try {
                    const prompt = buildPrompt(query.taskType || query.task || 'qa', query.subject || '', query.question || query.query_text);
                    
                    const runResults = await Promise.all(
                        modelIds.map(async (modelId) => {
                            const startTime = Date.now();
                            try {
                                logger.debug(`Calling model for query`, { modelId, queryIndex: i, run: run + 1 });
                                const answer = await callLLM(modelId, prompt);
                                const responseTime = Date.now() - startTime;
                                stats.byModel[modelId] = (stats.byModel[modelId] || 0) + 1;

                                const result = {
                                    modelId,
                                    model: MODEL_CONFIG.llm[modelId],
                                    answer,
                                    responseTime,
                                    success: true,
                                    run: run + 1,
                                    tokenCount: answer ? answer.split(/\s+/).length : 0
                                };

                                // Calculate evaluation metrics if reference is provided
                                if (query.reference || query.answer) {
                                    const reference = query.reference || query.answer;
                                    try {
                                        const metrics = evaluation.evaluateResponse(answer, reference);
                                        result.metrics = metrics;
                                        evaluationMetrics.push({ modelId, metrics, queryIndex: i });
                                    } catch (metricError) {
                                        logger.warn('Metric calculation failed', { modelId, error: metricError.message });
                                    }
                                }

                                return result;
                            } catch (error) {
                                logger.error(`Model call failed`, { modelId, queryIndex: i, error: error.message });
                                return {
                                    modelId,
                                    model: MODEL_CONFIG.llm[modelId],
                                    error: error.message,
                                    responseTime: Date.now() - startTime,
                                    success: false,
                                    run: run + 1
                                };
                            }
                        })
                    );

                    results.push({
                        query: {
                            ...query,
                            taskType: query.taskType || query.task || 'qa',
                            subject: query.subject || '',
                            question: query.question || query.query_text
                        },
                        results: runResults,
                        run: run + 1,
                        timestamp: new Date().toISOString()
                    });

                    // Rate limiting delay
                    if (i < queries.length - 1 || run < runs - 1) {
                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                } catch (error) {
                    logger.error('Query processing error', { queryIndex: i, run: run + 1, error: error.message });
                    results.push({
                        query,
                        error: error.message,
                        run: run + 1,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        }

        // Aggregate metrics by model
        const aggregatedMetrics = {};
        modelIds.forEach(modelId => {
            const modelMetrics = evaluationMetrics.filter(m => m.modelId === modelId);
            if (modelMetrics.length > 0) {
                aggregatedMetrics[modelId] = {
                    rouge: {
                        rouge1: modelMetrics.reduce((sum, m) => sum + (m.metrics.rouge?.rouge1 || 0), 0) / modelMetrics.length,
                        rouge2: modelMetrics.reduce((sum, m) => sum + (m.metrics.rouge?.rouge2 || 0), 0) / modelMetrics.length,
                        rougeL: modelMetrics.reduce((sum, m) => sum + (m.metrics.rouge?.rougeL || 0), 0) / modelMetrics.length
                    },
                    bleu: modelMetrics.reduce((sum, m) => sum + (m.metrics.bleu || 0), 0) / modelMetrics.length,
                    bertscore: {
                        f1: modelMetrics.reduce((sum, m) => sum + (m.metrics.bertscore?.f1 || 0), 0) / modelMetrics.length
                    },
                    overallScore: modelMetrics.reduce((sum, m) => sum + (m.metrics.overallScore || 0), 0) / modelMetrics.length
                };
            }
        });

        const totalTime = Date.now() - evaluationStartTime;
        logger.info(`Batch evaluation completed`, { 
            totalQueries: queries.length,
            totalResults: results.length,
            totalTime,
            successCount: results.filter(r => !r.error).length
        });

        res.json({
            success: true,
            results,
            metrics: aggregatedMetrics,
            summary: {
                totalQueries: queries.length,
                totalRuns: queries.length * runs,
                models: modelIds,
                completed: results.filter(r => !r.error).length,
                totalTime,
                averageTimePerQuery: totalTime / (queries.length * runs)
            }
        });

    } catch (error) {
        stats.errors++;
        logger.error('Batch Evaluation Error', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// =============================================================================
// API ROUTES - AUTO EVALUATION (LLM-as-Judge)
// =============================================================================

app.post('/api/auto-eval', async (req, res) => {
    stats.totalRequests++;

    try {
        const { response, query, reference, judgeModel = 'gemini-1.5-pro' } = req.body;

        if (!response) {
            return res.status(400).json({ success: false, error: 'Response is required' });
        }

        if (!query) {
            return res.status(400).json({ success: false, error: 'Query is required' });
        }

        if (!MODEL_CONFIG.llm[judgeModel]) {
            return res.status(400).json({ success: false, error: `Invalid judge model: ${judgeModel}` });
        }

        logger.info(`Auto-evaluating response with ${judgeModel}`, { 
            responseLength: response.length,
            hasReference: !!reference 
        });

        // Calculate automatic metrics if reference is provided
        let automaticMetrics = null;
        if (reference) {
            try {
                automaticMetrics = evaluation.evaluateResponse(response, reference);
                logger.debug('Automatic metrics calculated', { 
                    rouge1: automaticMetrics.rouge?.rouge1,
                    bleu: automaticMetrics.bleu,
                    overallScore: automaticMetrics.overallScore
                });
            } catch (metricError) {
                logger.warn('Automatic metric calculation failed', { error: metricError.message });
            }
        }

        // Build judging prompt
        const judgingPrompt = buildJudgingPrompt(response, query, reference);

        const startTime = Date.now();
        let judgeResponse = null;
        let llmScores = null;

        try {
            judgeResponse = await callLLM(judgeModel, judgingPrompt);
            const responseTime = Date.now() - startTime;

            // Parse scores from judge response
            llmScores = parseJudgeScores(judgeResponse);

            logger.debug('LLM-as-judge scores parsed', { scores: llmScores });
        } catch (judgeError) {
            logger.error('LLM-as-judge evaluation failed', { error: judgeError.message });
            // Continue with automatic metrics only
        }

        res.json({
            success: true,
            scores: llmScores || {},
            automaticMetrics: automaticMetrics || null,
            judgeResponse,
            responseTime: Date.now() - startTime,
            judgeModel,
            hasAutomaticMetrics: !!automaticMetrics,
            hasLLMScores: !!llmScores
        });

    } catch (error) {
        stats.errors++;
        logger.error('Auto-Eval Error', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

function buildJudgingPrompt(response, query, reference) {
    return `You are an expert evaluator for educational AI responses. Evaluate the following response based on the rubric below.

QUERY: ${query.query_text || query.question || query}
TASK TYPE: ${query.taskType || query.task || 'qa'}
SUBJECT: ${query.subject || 'General'}
${reference ? `REFERENCE ANSWER: ${reference}` : ''}

RESPONSE TO EVALUATE:
${response}

EVALUATION RUBRIC (Rate each criterion 1-5):
1. Accuracy: Factual correctness and precision (For STEM: prioritize precision; for humanities: prioritize nuance)
2. Completeness: Addresses all aspects of the query
3. Clarity: Easy to understand, well-structured
4. Relevance: Directly answers the query
5. Educational Value: Helps learning, provides context

Please provide scores in this exact JSON format:
{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "relevance": <1-5>,
  "educationalValue": <1-5>,
  "overall": <1-5>,
  "notes": "<brief explanation>"
}`;
}

function parseJudgeScores(judgeResponse) {
    try {
        // Try to extract JSON from response
        const jsonMatch = judgeResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            return {
                accuracy: parsed.accuracy || 3,
                completeness: parsed.completeness || 3,
                clarity: parsed.clarity || 3,
                relevance: parsed.relevance || 3,
                educationalValue: parsed.educationalValue || parsed.educational_value || 3,
                overall: parsed.overall || 3,
                notes: parsed.notes || ''
            };
        }
    } catch (err) {
        console.warn('Failed to parse judge scores:', err);
    }

    // Fallback: extract numbers from response
    const numbers = judgeResponse.match(/\b[1-5]\b/g);
    return {
        accuracy: numbers?.[0] ? parseInt(numbers[0]) : 3,
        completeness: numbers?.[1] ? parseInt(numbers[1]) : 3,
        clarity: numbers?.[2] ? parseInt(numbers[2]) : 3,
        relevance: numbers?.[3] ? parseInt(numbers[3]) : 3,
        educationalValue: numbers?.[4] ? parseInt(numbers[4]) : 3,
        overall: numbers?.[5] ? parseInt(numbers[5]) : 3,
        notes: 'Auto-parsed from response'
    };
}

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'test-compare.html'));
});

// =============================================================================
// ERROR HANDLING MIDDLEWARE
// =============================================================================

app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ success: false, error: 'File too large. Maximum 20MB.' });
        }
        return res.status(400).json({ success: false, error: `Upload error: ${error.message}` });
    }

    if (error.message?.includes('Invalid file type')) {
        return res.status(400).json({ success: false, error: error.message });
    }

    console.error('Unhandled error:', error);
    stats.errors++;
    res.status(500).json({ success: false, error: 'Internal server error' });
});

// =============================================================================
// SERVER STARTUP
// =============================================================================

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════════╗
║        Educational AI Testing Platform - Server Started          ║
╠══════════════════════════════════════════════════════════════════╣
║  🌐 URL: http://localhost:${PORT}                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  📡 LLM Endpoints:                                               ║
║     POST /api/ask          - Single LLM query                    ║
║     POST /api/compare      - Compare multiple LLMs               ║
║                                                                  ║
║  📄 OCR Endpoints:                                               ║
║     POST /api/ocr          - Single OCR extraction               ║
║     POST /api/ocr-compare  - Compare multiple OCR modes          ║
║                                                                  ║
║  🔧 Utility:                                                     ║
║     GET  /api/models       - List available models               ║
║     GET  /api/stats        - View usage statistics               ║
║     GET  /api/health       - Health check                        ║
╠══════════════════════════════════════════════════════════════════╣
║  🔑 Provider Status:                                             ║
║     OpenRouter: ${OPENROUTER_API_KEY ? '✅ Ready' : '❌ Not configured'}                                ║
║     Groq:       ${GROQ_API_KEY ? '✅ Ready' : '❌ Not configured'}                                ║
║     Gemini:     ${GEMINI_API_KEY ? '✅ Ready' : '❌ Not configured'}                                ║
╠══════════════════════════════════════════════════════════════════╣
║  📁 Supported Files:                                             ║
║     Images: JPG, PNG, GIF, BMP, WebP, TIFF                       ║
║     Documents: PDF (converted to images)                         ║
╚══════════════════════════════════════════════════════════════════╝
    `);
});

module.exports = app;
