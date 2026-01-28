# 🚀 Educational AI Testing Platform

<div align="center">

![Node.js](https://img.shields.io/badge/Node.js-18+-green?style=for-the-badge&logo=node.js)
![Express.js](https://img.shields.io/badge/Express.js-4.22-blue?style=for-the-badge&logo=express)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?style=for-the-badge&logo=javascript)
![AI/ML](https://img.shields.io/badge/AI%2FML-Multiple%20Providers-orange?style=for-the-badge&logo=tensorflow)

**A comprehensive full-stack platform for testing and comparing Large Language Models (LLMs) and OCR solutions**

[Live Demo](#) • [Documentation](#api-documentation) • [Report Issue](#)

</div>

---

## 👨‍💻 About the Developer

**Mohamed Salman** | Full-Stack Developer & AI/ML Engineer

Passionate about building innovative solutions that bridge AI technology with practical applications. This project demonstrates expertise in full-stack development, API integration, and AI model evaluation.

🔗 **Connect with me:**
- GitHub: [@mohameddsalmann](https://github.com/mohameddsalmann)
- Portfolio: [Your Portfolio Link]
- LinkedIn: [Your LinkedIn Profile]

---

## ✨ Key Highlights & Skills Demonstrated

### 🎯 Technical Excellence
- ✅ **Full-Stack Development**: Built end-to-end solution with Node.js/Express backend and modern vanilla JavaScript frontend
- ✅ **API Integration**: Successfully integrated multiple AI providers (Google Gemini, OpenRouter, Groq, HuggingFace)
- ✅ **OCR Processing**: Implemented advanced Tesseract.js configurations for document text extraction
- ✅ **Performance Optimization**: Parallel processing, efficient file handling, and real-time statistics tracking
- ✅ **Modern Architecture**: RESTful API design, modular code structure, comprehensive error handling

### 💼 Business Value
- 📊 **Decision Support**: Enables data-driven selection of AI models for educational applications
- ⚡ **Cost Optimization**: Compare multiple providers to find the best price-performance ratio
- 🎓 **Educational Impact**: Helps educators choose the right AI tools for their specific needs
- 🔄 **Scalable Solution**: Designed for extensibility with easy addition of new models and providers

### 🛠 Core Competencies Showcased
- **Backend Development**: Express.js, REST APIs, File Upload Handling, Middleware Development
- **Frontend Development**: HTML5, CSS3, Vanilla JavaScript, Responsive Design, UX/UI
- **AI/ML Integration**: LLM APIs, OCR Processing, Model Evaluation, Performance Benchmarking
- **DevOps**: Environment Configuration, Error Handling, Health Monitoring, Logging
- **Problem Solving**: Complex multi-provider integration, performance optimization, user experience design

## 🎯 Project Overview

This platform provides a **production-ready Proof of Concept** that demonstrates advanced full-stack development skills and AI integration expertise. Built from scratch, it showcases:

- **Real-world Problem Solving**: Addresses the challenge of selecting optimal AI models for educational use cases
- **Multi-Provider Architecture**: Seamlessly integrates 4+ AI service providers with unified interface
- **Performance Benchmarking**: Built-in analytics and comparison tools for data-driven decisions
- **User-Centric Design**: Intuitive UI with drag-and-drop functionality and real-time feedback

### 🏆 What Makes This Project Stand Out

1. **Comprehensive Integration**: Successfully integrated multiple AI providers with different authentication methods and API structures
2. **Production-Ready Code**: Includes error handling, health checks, statistics tracking, and proper file management
3. **Extensible Architecture**: Easy to add new models, providers, or features without major refactoring
4. **Documentation Excellence**: Complete API documentation, troubleshooting guides, and clear code structure

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Detailed Overview

This platform provides a unified interface for:

- **LLM Testing**: Compare responses from multiple language models (Mistral, Llama, Gemini, etc.) for educational tasks like Q&A, summarization, and concept explanations
- **OCR Testing**: Test different Tesseract OCR configurations to find the best setup for document text extraction
- **Performance Analysis**: Track response times, accuracy metrics, and generate comparison statistics

### Real-World Use Cases

- ✅ **Educational Institutions**: Evaluate which LLM performs best for specific educational subjects
- ✅ **Document Processing**: Compare OCR accuracy across different document types
- ✅ **AI Research**: Benchmark model performance and response times
- ✅ **Prompt Engineering**: Test and optimize prompt strategies
- ✅ **Cost Analysis**: Analyze OCR configuration effectiveness and provider costs

## ✨ Features

### LLM Testing
- **Multiple Providers**: Support for OpenRouter, Groq, Google Gemini, and HuggingFace
- **Task Types**: 
  - Question Answering (Q&A)
  - Text Summarization
  - Concept Explanation
- **Subject Areas**: Mathematics, Physics, Chemistry, Biology, Computer Science, Programming, History, Geography, Literature, Economics, Philosophy
- **Comparison Mode**: Compare up to 5 models simultaneously
- **Performance Metrics**: Response time, answer quality, character count

### OCR Testing
- **Multiple Configurations**: 10+ Tesseract OCR configurations
- **File Support**: JPG, PNG, GIF, BMP, WebP, TIFF (max 10MB)
- **OCR Modes**: 
  - Default, Accurate, Fast
  - Single Block, Single Line, Single Word
  - Sparse Text, Multilingual, Column-based
- **Metrics**: Confidence score, word count, line count, paragraph count

### Platform Features
- **Modern UI**: Responsive design with drag-and-drop file upload
- **Real-time Statistics**: Track usage, performance, and model popularity
- **Export Results**: Download comparison results as JSON
- **Health Monitoring**: API health checks and provider status
- **Error Handling**: Comprehensive error messages and recovery

## 🛠 Technology Stack

### Backend
- **Node.js**: Runtime environment
- **Express.js**: Web framework
- **Multer**: File upload handling
- **Tesseract.js**: OCR processing
- **Axios**: HTTP client for API calls
- **dotenv**: Environment variable management
- **CORS**: Cross-origin resource sharing

### Frontend
- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No framework dependencies
- **Google Fonts (Inter)**: Typography

### AI/ML Services
- **Google Generative AI**: Gemini models
- **OpenRouter**: Mistral and other models
- **Groq**: Fast inference for Llama models
- **HuggingFace**: Open-source model inference

## 📦 Prerequisites

Before installing, ensure you have:

- **Node.js** (v14 or higher)
- **npm** (v6 or higher)
- **API Keys** for at least one provider:
  - OpenRouter API key (optional)
  - Groq API key (optional)
  - Google Gemini API key (optional)
  - HuggingFace API key (optional)

> **Note**: You don't need all API keys. The platform works with any combination of configured providers.

## 🚀 Installation

### Step 1: Clone or Navigate to Project

```bash
cd work
```

### Step 2: Install Dependencies

```bash
npm install
```

This will install all required packages including:
- Express and server dependencies
- Tesseract.js for OCR
- AI provider SDKs
- File upload middleware

### Step 3: Environment Configuration

Create a `.env` file in the `work` directory:

```env
# API Keys (add at least one)
OPENROUTER_API_KEY=your_openrouter_key_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
HF_API_KEY=your_huggingface_key_here

# Server Configuration (optional)
PORT=3000
```

### Step 4: Verify Installation

Check that the `uploads` directory exists (it will be created automatically if missing).

## ⚙️ Configuration

### API Keys Setup

#### OpenRouter
1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up and create an API key
3. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

#### Groq
1. Visit [Groq.com](https://groq.com)
2. Create an account and generate API key
3. Add to `.env`: `GROQ_API_KEY=gsk_...`

#### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env`: `GEMINI_API_KEY=AIza...`

#### HuggingFace
1. Visit [HuggingFace](https://huggingface.co)
2. Go to Settings > Access Tokens
3. Create a new token
4. Add to `.env`: `HF_API_KEY=hf_...`

### Port Configuration

Default port is `3000`. To change:

```env
PORT=8080
```

## 🎮 Usage

### Starting the Server

**Development mode** (with auto-reload):
```bash
npm run dev
```

**Production mode**:
```bash
npm start
```

The server will start on `http://localhost:3000` (or your configured port).

### Accessing the Platform

1. Open your browser and navigate to `http://localhost:3000`
2. You'll see the main interface with two modes:
   - **LLM Testing**: For language model comparisons
   - **OCR Testing**: For document text extraction

### LLM Testing Workflow

1. **Select Mode**: Choose "LLM Testing" tab
2. **Choose Comparison**: Single model or Compare models (2-5)
3. **Configure Task**:
   - Select task type (Q&A, Summarize, Explain)
   - Choose subject area (optional)
4. **Enter Query**: Type your question or text
5. **Select Model(s)**: Choose from available models
6. **Submit**: Click "Get Result" or "Compare Models"
7. **View Results**: See responses with performance metrics

### OCR Testing Workflow

1. **Select Mode**: Choose "OCR Testing" tab
2. **Choose Comparison**: Single OCR config or Compare configs (2-5)
3. **Upload Document**: Drag & drop or click to browse
4. **Select OCR Config(s)**: Choose from available configurations
5. **Submit**: Click "Get Result" or "Compare Configs"
6. **View Results**: See extracted text with confidence scores

### Understanding Results

**LLM Results Include**:
- Model response/answer
- Response time (seconds)
- Character count
- Model provider information

**OCR Results Include**:
- Extracted text
- Confidence score (percentage)
- Word count
- Line count
- Paragraph count
- Processing time

**Comparison Mode**:
- Side-by-side results
- Fastest response highlighted
- Performance statistics
- Export to JSON option

## 📡 API Documentation

### Base URL
```
http://localhost:3000
```

### LLM Endpoints

#### Single LLM Query
```http
POST /api/ask
Content-Type: application/json

{
  "question": "What is photosynthesis?",
  "subject": "biology",
  "taskType": "qa",
  "modelId": "gemini-1.5-flash"
}
```

**Response**:
```json
{
  "success": true,
  "answer": "Photosynthesis is...",
  "responseTime": 1234,
  "model": { ... },
  "taskType": "qa",
  "subject": "biology"
}
```

#### Compare Multiple LLMs
```http
POST /api/compare
Content-Type: application/json

{
  "question": "Explain quantum mechanics",
  "subject": "physics",
  "taskType": "explain",
  "modelIds": ["gemini-1.5-flash", "llama-3.3-70b-versatile"]
}
```

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "modelId": "gemini-1.5-flash",
      "model": { ... },
      "answer": "...",
      "responseTime": 1234,
      "success": true
    },
    ...
  ],
  "query": { ... }
}
```

### OCR Endpoints

#### Single OCR Extraction
```http
POST /api/ocr
Content-Type: multipart/form-data

document: [file]
modelId: "tesseract-default"
```

**Response**:
```json
{
  "success": true,
  "text": "Extracted text...",
  "confidence": 95.5,
  "words": 150,
  "lines": 10,
  "paragraphs": 3,
  "responseTime": 2345,
  "model": { ... },
  "fileName": "document.jpg"
}
```

#### Compare OCR Configs
```http
POST /api/ocr-compare
Content-Type: multipart/form-data

document: [file]
modelIds: ["tesseract-default", "tesseract-accurate"]
```

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "modelId": "tesseract-default",
      "model": { ... },
      "text": "...",
      "confidence": 95.5,
      "words": 150,
      "responseTime": 2345,
      "success": true
    },
    ...
  ],
  "fileName": "document.jpg"
}
```

### Utility Endpoints

#### List Available Models
```http
GET /api/models
```

**Response**:
```json
{
  "success": true,
  "models": {
    "llm": [ ... ],
    "ocr": [ ... ]
  }
}
```

#### Get Statistics
```http
GET /api/stats
```

**Response**:
```json
{
  "success": true,
  "stats": {
    "totalRequests": 150,
    "llmRequests": 100,
    "ocrRequests": 50,
    "comparisons": 30,
    "byModel": { ... },
    "errors": 2,
    "uptime": 3600,
    "memoryUsage": { ... }
  }
}
```

#### Health Check
```http
GET /api/health
```

**Response**:
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "providers": {
    "openrouter": true,
    "groq": true,
    "gemini": true,
    "huggingface": false
  }
}
```

## 📁 Project Structure

```
work/
├── server.js                 # Main Express server
├── test-compare.html         # Frontend interface
├── package.json             # Dependencies and scripts
├── package-lock.json        # Locked dependency versions
├── .env                     # Environment variables (create this)
├── README.md                # This file
├── uploads/                 # Temporary file storage (auto-created)
│   └── [temporary files]
├── eng.traineddata          # Tesseract language data
└── node_modules/            # Dependencies (auto-created)
```

### Key Files

- **server.js**: Contains all backend logic:
  - API route handlers
  - LLM provider integrations
  - OCR processing functions
  - Statistics tracking
  - Error handling middleware

- **test-compare.html**: Single-page application with:
  - Mode switching (LLM/OCR)
  - Model selection interface
  - Results display
  - File upload handling
  - API integration

## 🤖 Supported Models

### LLM Models

| Model ID | Name | Provider | Cost | Speed | Quality |
|----------|------|----------|------|-------|---------|
| `mistral-7b-instruct` | Mistral 7B Instruct | OpenRouter | Free | Fast | ⭐⭐⭐⭐ |
| `llama-3.1-8b-instant` | Llama 3.1 8B Instant | Groq | Free | Fastest | ⭐⭐⭐⭐ |
| `llama-3.3-70b-versatile` | Llama 3.3 70B Versatile | Groq | Free | Very Fast | ⭐⭐⭐⭐⭐ |
| `gemma2-9b-it` | Gemma 2 9B | Groq | Free | Fast | ⭐⭐⭐⭐ |
| `mixtral-8x7b` | Mixtral 8x7B | Groq | Free | Fast | ⭐⭐⭐⭐ |
| `gemini-1.5-flash` | Gemini 1.5 Flash | Google | Cheap | Fast | ⭐⭐⭐⭐⭐ |
| `gemini-2.0-flash` | Gemini 2.0 Flash | Google | Cheap | Very Fast | ⭐⭐⭐⭐⭐ |
| `gemini-pro` | Gemini Pro | Google | Cheap | Medium | ⭐⭐⭐⭐⭐ |

### OCR Configurations

| Config ID | Name | Best For | Speed | Quality |
|-----------|------|----------|-------|---------|
| `tesseract-default` | Tesseract Default | General purpose | Medium | ⭐⭐⭐⭐ |
| `tesseract-accurate` | Tesseract Accurate | High accuracy | Medium | ⭐⭐⭐⭐⭐ |
| `tesseract-fast` | Tesseract Fast | Quick processing | Fast | ⭐⭐⭐ |
| `tesseract-single-block` | Single Block | Uniform text blocks | Fast | ⭐⭐⭐⭐ |
| `tesseract-single-line` | Single Line | Headers, titles | Fastest | ⭐⭐⭐⭐ |
| `tesseract-single-word` | Single Word | Individual words | Fastest | ⭐⭐⭐⭐ |
| `tesseract-sparse-text` | Sparse Text | Scattered text | Medium | ⭐⭐⭐ |
| `tesseract-multilingual` | Multilingual | Multi-language docs | Slow | ⭐⭐⭐⭐ |
| `tesseract-column` | Column | Newspaper columns | Medium | ⭐⭐⭐⭐ |
| `tesseract-raw-line` | Raw Line | Single line detection | Fast | ⭐⭐⭐ |

## 🔧 Development

### Development Scripts

```bash
# Start with auto-reload (nodemon)
npm run dev

# Start production server
npm start

# Run tests (when implemented)
npm test
```

### Adding New Models

To add a new LLM model, edit `server.js` and add to `MODEL_CONFIG.llm`:

```javascript
'new-model-id': {
    provider: 'provider-name',
    modelId: 'actual-model-id',
    name: 'Display Name',
    icon: '🎯',
    cost: 'free',
    speed: 'fast',
    quality: 5
}
```

Then add the corresponding provider function if needed.

### Adding New OCR Configs

Add to `MODEL_CONFIG.ocr` in `server.js`:

```javascript
'new-ocr-config': {
    name: 'Display Name',
    icon: '📄',
    lang: 'eng',
    oem: 3,
    psm: 3,
    description: 'Description',
    bestFor: 'Use case'
}
```

### Code Structure

- **Modular Design**: Functions organized by purpose (LLM, OCR, utilities)
- **Error Handling**: Comprehensive try-catch blocks with user-friendly messages
- **Statistics Tracking**: Built-in usage analytics
- **File Management**: Automatic cleanup of uploaded files

## 🐛 Troubleshooting

### Common Issues

#### "API key not configured"
- **Solution**: Ensure your `.env` file exists and contains the required API keys
- Check that keys are correctly formatted (no extra spaces)

#### "Port already in use"
- **Solution**: Change the `PORT` in `.env` or stop the process using port 3000
- On Windows: `netstat -ano | findstr :3000` then `taskkill /PID <pid> /F`

#### "OCR processing failed"
- **Solution**: Ensure `eng.traineddata` is in the project directory
- Check file format is supported (JPG, PNG, GIF, BMP, WebP, TIFF)
- Verify file size is under 10MB

#### "Gemini quota exceeded"
- **Solution**: Wait a few minutes and try again, or use a different model
- Check your Google Cloud quota limits

#### "File upload fails"
- **Solution**: Ensure `uploads/` directory exists and is writable
- Check file size is under 10MB
- Verify file type is supported

### Debug Mode

Enable verbose logging by checking console output. The server logs:
- Request processing
- Model selection
- Response times
- Errors with stack traces

### Health Check

Visit `http://localhost:3000/api/health` to verify:
- Server status
- Provider API key configuration
- System health

## 📊 Performance Considerations

- **File Uploads**: Automatically deleted after processing
- **OCR Processing**: Sequential processing to avoid memory issues
- **LLM Requests**: Parallel processing for comparisons (up to 5 models)
- **Rate Limits**: Be aware of provider rate limits (especially Gemini)
- **Memory Usage**: Monitor with `/api/stats` endpoint

## 🔒 Security Notes

- **API Keys**: Never commit `.env` file to version control
- **File Uploads**: Files are validated and automatically cleaned up
- **CORS**: Currently enabled for all origins (restrict in production)
- **Input Validation**: All inputs are validated before processing

## 🤝 Contributing

When contributing to this project:

1. Follow existing code style
2. Add comments for complex logic
3. Test with multiple models/configs
4. Update documentation for new features
5. Ensure error handling is comprehensive

## 📝 License

This is a POC (Proof of Concept) project demonstrating full-stack development and AI integration capabilities.

## 📞 Contact & Collaboration

I'm open to:
- 💼 **Job Opportunities**: Full-stack, Backend, Frontend, or AI/ML Engineering roles
- 🤝 **Collaborations**: Open to contributing to interesting projects
- 📧 **Questions**: Feel free to reach out via GitHub Issues or direct message

For technical issues:
1. Check the Troubleshooting section
2. Review API documentation
3. Check server logs for error details
4. Open an issue on GitHub

## 🎯 Future Enhancements

Planned improvements to showcase continued learning:
- [ ] User authentication and session management
- [ ] Database integration for storing comparison history
- [ ] Advanced analytics dashboard with visualizations
- [ ] Multi-language OCR support expansion
- [ ] Batch processing capabilities
- [ ] API rate limiting and security enhancements
- [ ] Docker containerization for easy deployment
- [ ] Comprehensive unit and integration test suite
- [ ] CI/CD pipeline implementation

---

## 📊 Project Statistics

- **Lines of Code**: ~2000+ (Backend + Frontend)
- **API Endpoints**: 6+ RESTful endpoints
- **Supported Models**: 8+ LLM models, 10+ OCR configurations
- **AI Providers**: 4 major providers integrated
- **Development Time**: Built as a comprehensive POC demonstrating production-level skills

---

<div align="center">

**Built with ❤️ by Mohamed Salman**

**Version**: 2.0.0 | **Last Updated**: 2024

⭐ **Star this repo if you find it helpful!**

[⬆ Back to Top](#-educational-ai-testing-platform)

</div>

