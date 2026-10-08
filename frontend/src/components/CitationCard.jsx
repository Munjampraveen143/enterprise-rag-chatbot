function CitationCard({
  citation,
  index,
}) {
  return (
    <div className="citation-card">
      <div className="citation-number">
        S{index}
      </div>

      <div className="citation-details">
        <div className="citation-document">
          {citation.document ||
            "Unknown document"}
        </div>

        <div className="citation-meta">
          {citation.page && (
            <span>
              Page {citation.page}
            </span>
          )}

          {citation.section && (
            <span>
              {citation.section}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default CitationCard;