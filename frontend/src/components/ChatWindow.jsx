import { useEffect, useRef, useState } from "react";

function ChatWindow({
  messages,
  loading,
  onSend,
}) {
  const [question, setQuestion] = useState("");

  const inputRef = useRef(null);
  const messagesRef = useRef(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop =
        messagesRef.current.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(() => {
    if (!loading) {
      inputRef.current?.focus();
    }
  }, [loading]);

  function handleSend() {
    const trimmed =
      question.trim();

    if (!trimmed || loading) {
      return;
    }

    onSend(trimmed);

    setQuestion("");
  }

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      handleSend();
    }
  }

  function renderMessage(
    message,
    index
  ) {
    if (
      message.role === "user"
    ) {
      return (
        <div
          className="message user"
          key={index}
        >
          <div className="user-message">
            {message.content}
          </div>
        </div>
      );
    }

    return (
      <div
        className="message assistant"
        key={index}
      >
        <div className="ai-message">

          <div className="ai-label">
            <div className="ai-icon">
              ✦
            </div>

            <span>
              AI Assistant
            </span>
          </div>

          <div className="answer-text">
            {message.content}
          </div>

          {message.citations &&
            message.citations.length >
              0 && (
              <div className="sources">

                <div className="sources-title">
                  Sources
                </div>

                {message.citations.map(
                  (
                    citation,
                    citationIndex
                  ) => (
                    <div
                      className="source-item"
                      key={
                        citationIndex
                      }
                    >
                      <span className="source-id">
                        S
                        {citationIndex +
                          1}
                      </span>

                      <span>
                        {citation.document ||
                          "Document"}

                        {citation.page
                          ? ` • Page ${citation.page}`
                          : ""}

                        {citation.section
                          ? ` • ${citation.section}`
                          : ""}
                      </span>
                    </div>
                  )
                )}
              </div>
            )}
        </div>
      </div>
    );
  }

  return (
    <main className="chat-panel">

      {/* =========================
          CHAT HEADER
      ========================= */}

      <div className="chat-header">

        <div className="chat-title">
          AI Knowledge Assistant
        </div>

        <div className="status">
          <span className="status-dot" />
          AI Online
        </div>

      </div>


      {/* =========================
          MESSAGES
      ========================= */}

      <div
        className="messages"
        ref={messagesRef}
      >

        {messages.length === 0 ? (
          <div className="empty-chat">

            <div className="empty-chat-box">

              <div className="empty-icon">
                ✦
              </div>

              <h2>
                Ask your documents
              </h2>

              <p>
                Ask questions about your
                uploaded documents and get
                answers using your enterprise
                knowledge base.
              </p>

            </div>

          </div>
        ) : (
          messages.map(
            renderMessage
          )
        )}


        {/* =========================
            AI THINKING
        ========================= */}

        {loading && (
          <div className="message assistant">

            <div className="ai-message">

              <div className="ai-label">

                <div className="ai-icon">
                  ✦
                </div>

                <span>
                  AI Assistant
                </span>

              </div>

              <div className="loading">

                <span>
                  Thinking
                </span>

                <span className="loading-dot" />
                <span className="loading-dot" />
                <span className="loading-dot" />

              </div>

            </div>

          </div>
        )}

      </div>


      {/* =========================
          CHATGPT STYLE COMPOSER
      ========================= */}

      <div className="input-area">

        <textarea
          ref={inputRef}
          className="chat-input"

          value={question}

          onChange={(event) =>
            setQuestion(
              event.target.value
            )
          }

          onKeyDown={
            handleKeyDown
          }

          placeholder="Message AI Knowledge Assistant..."

          rows={1}

          disabled={loading}
        />

        <button
          className="send-button"

          onClick={handleSend}

          disabled={
            loading ||
            !question.trim()
          }

          aria-label="Send message"

          title="Send message"
        >
          ↑
        </button>

      </div>

    </main>
  );
}

export default ChatWindow;