import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [sessionId] = useState(() => {
    const existing = sessionStorage.getItem(
      "rag_session_id"
    );

    if (existing) {
      return existing;
    }

    const newId = crypto.randomUUID();

    sessionStorage.setItem(
      "rag_session_id",
      newId
    );

    return newId;
  });

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const response = await fetch(
        `${API_BASE}/documents`
      );

      const data = await response.json();

      console.log(
        "Documents response:",
        data
      );

      if (!response.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Documents request failed: ${response.status}`
        );
      }

      const documentList = Array.isArray(data)
        ? data
        : data.documents || [];

      setDocuments(documentList);

    } catch (error) {
      console.error(
        "Failed to load documents:",
        error
      );
    }
  }

  async function sendMessage(text) {
    if (!text.trim() || loading) {
      return;
    }

    const question = text.trim();

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: question,
      },
    ]);

    setLoading(true);

    try {
      console.log(
        "Sending question:",
        question
      );

      const response = await fetch(
        `${API_BASE}/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            session_id: sessionId,
            message: question,
            top_k: 5,
          }),
        }
      );

      const data = await response.json();

      console.log(
        "CHAT STATUS:",
        response.status
      );

      console.log(
        "CHAT RESPONSE:",
        data
      );

      if (!response.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Chat request failed: ${response.status}`
        );
      }

      const answer =
        data.answer ||
        "The information is not available in the provided documents.";

      const citations =
        Array.isArray(data.citations)
          ? data.citations
          : [];

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          citations: citations,
        },
      ]);

    } catch (error) {
      console.error(
        "CHAT ERROR:",
        error
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",

          content:
            `Sorry, I could not process your question.\n\n` +
            `${error.message}`,

          citations: [],
        },
      ]);

    } finally {
      setLoading(false);
    }
  }

  async function uploadDocument(file) {
    if (!file) {
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      console.log(
        "Uploading:",
        file.name
      );

      const response = await fetch(
        `${API_BASE}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      console.log(
        "Upload response:",
        data
      );

      if (!response.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Upload failed: ${response.status}`
        );
      }

      console.log(
        "Upload successful:",
        data
      );

      await loadDocuments();

    } catch (error) {
      console.error(
        "Upload failed:",
        error
      );

      alert(
        `Upload failed: ${error.message}`
      );

    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId) {
    try {
      const response = await fetch(
        `${API_BASE}/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json();

      console.log(
        "Delete response:",
        data
      );

      if (!response.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Delete failed: ${response.status}`
        );
      }

      setDocuments((prev) =>
        prev.filter(
          (document) =>
            document.id !== documentId
        )
      );

    } catch (error) {
      console.error(
        "Delete failed:",
        error
      );

      alert(
        `Delete failed: ${error.message}`
      );
    }
  }

  return (
    <div className="app-shell">

      <Sidebar
        documents={documents}
        onUpload={uploadDocument}
        onDelete={deleteDocument}
        uploading={uploading}
      />

      <ChatWindow
        messages={messages}
        loading={loading}
        onSend={sendMessage}
      />

    </div>
  );
}

export default App;
