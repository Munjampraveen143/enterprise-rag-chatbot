import { useRef } from "react";

function DocumentList({
  documents = [],
  onUpload,
  onDelete,
  uploading = false,
}) {
  const fileInputRef = useRef(null);

  function handleUploadClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    console.log("Selected file:", file);
    console.log("File name:", file.name);
    console.log("File type:", file.type);
    console.log("File size:", file.size);

    onUpload(file);

    // Allows selecting the same file again later.
    event.target.value = "";
  }

  return (
    <div className="document-list">
      <div className="document-list-header">
        <div>
          <h3>Documents</h3>
          <span>{documents.length} uploaded</span>
        </div>

        <button
          type="button"
          className="upload-button"
          onClick={handleUploadClick}
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload"}
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>

      <div className="documents">
        {documents.length === 0 ? (
          <div className="empty-documents">
            No documents uploaded yet.
          </div>
        ) : (
          documents.map((document) => (
            <div
              key={document.id}
              className="document-card"
            >
              <div className="document-info">
                <div className="document-name">
                  {document.filename}
                </div>

                <div className="document-meta">
                  {document.chunk_count ?? 0} chunks
                </div>
              </div>

              <button
                type="button"
                className="delete-button"
                onClick={() => onDelete(document.id)}
                title="Delete document"
              >
                Delete
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default DocumentList;