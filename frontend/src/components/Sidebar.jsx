import DocumentList from "./DocumentList";

function Sidebar({
  documents,
  onUpload,
  onDelete,
  uploading,
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <h1>AI Knowledge Assistant</h1>
        <p>Enterprise RAG</p>
      </div>

      <DocumentList
        documents={documents}
        onUpload={onUpload}
        onDelete={onDelete}
        uploading={uploading}
      />
    </aside>
  );
}

export default Sidebar;