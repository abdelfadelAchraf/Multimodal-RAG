
import { useState } from "react";
import axios from "axios";
import api from "./services/api";
import {
  FileText,
  Upload,
  Send,
  Loader2,
  MessageSquare,
  CheckCircle,
} from "lucide-react";
import "./App.css";

interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  page_count: number;
  chunks_count: number;
  images_count: number;
}

interface ChatResponse {
  question: string;
  answer: string;
  text_sources: {
    filename: string;
    page_number: number;
    text: string;
  }[];
  image_sources: {
    filename: string;
    page_number: number;
  }[];
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] = useState<UploadResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      setError("");
      setDocument(null);
      setResponse(null);

      const result = await api.post<UploadResponse>(
        "/documents/upload",
        formData
      );

      setDocument(result.data);
    } catch (err) {
      setError(
        axios.isAxiosError(err)
          ? err.response?.data?.detail || "Document upload failed."
          : "An unexpected error occurred."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !document) return;

    try {
      setAsking(true);
      setError("");
      setResponse(null);

      const result = await api.post<ChatResponse>("/chat", {
        question,
        top_k: 5,
        document_id: document.document_id,
      });

      setResponse(result.data);
    } catch (err) {
      setError(
        axios.isAxiosError(err)
          ? err.response?.data?.detail || "Unable to generate an answer."
          : "An unexpected error occurred."
      );
    } finally {
      setAsking(false);
    }
  };

  return (
    <main className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <MessageSquare size={22} />
          </div>
          <h1>OmniKnowledge</h1>
        </div>
        <span className="header-label">AI Document Assistant</span>
      </header>

      <section className="container">
        <div className="intro">
          <h2>Your documents, understood.</h2>
          <p>
            Upload a document and ask questions. Get answers based
            on the information inside your files.
          </p>
        </div>

        <section className="panel">
          <h3>
            <Upload size={19} />
            Upload a document
          </h3>

          <label className="upload-area">
            <FileText size={32} />
            <span>
              {file ? file.name : "Choose a PDF document"}
            </span>
            <small>PDF files only</small>

            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={(event) =>
                setFile(event.target.files?.[0] ?? null)
              }
            />
          </label>

          <button
            className="primary-button"
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? (
              <>
                <Loader2 className="spinner" size={18} />
                Processing document...
              </>
            ) : (
              "Upload PDF"
            )}
          </button>

          {document && (
            <div className="success-message">
              <CheckCircle size={18} />
              <div>
                <strong>Document ready!</strong>
                <p>
                  {document.filename} · {document.page_count} pages
                </p>
              </div>
            </div>
          )}
        </section>

        <section className="panel">
          <h3>
            <MessageSquare size={19} />
            Ask your document
          </h3>

          <textarea
            placeholder={
              document
                ? "Ask anything about your document..."
                : "Upload a document first..."
            }
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={!document || asking}
          />

          <button
            className="primary-button"
            onClick={handleAsk}
            disabled={!document || !question.trim() || asking}
          >
            {asking ? (
              <>
                <Loader2 className="spinner" size={18} />
                Generating answer...
              </>
            ) : (
              <>
                Ask question
                <Send size={17} />
              </>
            )}
          </button>
        </section>

        {error && <div className="error-message">{error}</div>}

        {response && (
          <section className="panel answer-panel">
            <h3>Answer</h3>
            <p className="answer-text">{response.answer}</p>

            <div className="sources">
              <h4>Sources</h4>

              {response.text_sources.map((source, index) => (
                <div className="source-item" key={`text-${index}`}>
                  <FileText size={17} />
                  <span>
                    {source.filename} — Page {source.page_number}
                  </span>
                </div>
              ))}

              {response.image_sources.map((source, index) => (
                <div className="source-item" key={`image-${index}`}>
                  <FileText size={17} />
                  <span>
                    {source.filename} — Image, Page {source.page_number}
                  </span>
                </div>
              ))}

              {response.text_sources.length === 0 &&
                response.image_sources.length === 0 && (
                  <p>No sources available.</p>
                )}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;
