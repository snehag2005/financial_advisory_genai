
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF file first.");

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post("http://127.0.0.1:8000/upload", formData);
      alert("File uploaded successfully");
    } catch (err) {
      alert("Upload failed");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question) return alert("Enter your question");

    setAsking(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        question: question,
      });
      setAnswer(res.data.answer);
    } catch (err) {
      alert("Failed to get answer");
      console.error(err);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="App">
      <h1>FinGPT Advisor</h1>
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={uploading}>
        {uploading ? "Uploading..." : "Upload PDF"}
      </button>

      <br /><br />

      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question from the uploaded PDF"
      />
      <button onClick={handleAsk} disabled={asking}>
        {asking ? "Asking..." : "Ask"}
      </button>

      <div className="answer">
        <h3>Answer:</h3>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
