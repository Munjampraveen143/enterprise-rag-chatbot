import CitationCard from "./CitationCard";

function MessageBubble({ message }) {
  const isUser =
    message.role === "user";

  return (
    <div
      className={`message-row ${
        isUser
          ? "user-row"
          : "assistant-row"
      }`}
    >
      <div
        className={`message-bubble ${
          isUser
            ? "user-message"
            : "assistant-message"
        }`}
      >
        <div className="message-role">
          {isUser
            ? "You"
            : "AI Assistant"}
        </div>

        <div className="message-content">
          {message.content}
        </div>

        {!isUser &&
          message.citations?.length > 0 && (
            <div className="citations-section">
              <div className="citations-title">
                Sources
              </div>

              {message.citations.map(
                (citation, index) => (
                  <CitationCard
                    key={`${citation.document_id}-${citation.chunk_id}-${index}`}
                    citation={citation}
                    index={index + 1}
                  />
                )
              )}
            </div>
          )}
      </div>
    </div>
  );
}

export default MessageBubble;