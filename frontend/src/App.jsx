import React, { useState } from "react";
import axios from "axios";
import ChatBox from "./components/ChatBox";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [act, setAct] = useState("ROB");

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const res = await axios.post("https://your-render-url.onrender.com/ask", {
        question: input,
        act_name: act,
      });
      const botMsg = { sender: "bot", text: res.data.answer };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [...prev, { sender: "bot", text: "Error fetching response." }]);
    }
    setInput("");
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-50 p-6">
      <h1 className="text-3xl font-bold mb-4">SSM Intelligence Hub</h1>
      <div className="mb-4">
        <label className="mr-2">Select Act:</label>
        <select value={act} onChange={(e) => setAct(e.target.value)}>
          <option value="ROB">ROB Act 1956</option>
          <option value="ROC">ROC Act 2016</option>
          <option value="LLP">LLP Act 2024</option>
        </select>
      </div>
      <ChatBox messages={messages} />
      <div className="mt-4 flex">
        <input
          className="border px-4 py-2 rounded-l-md w-80"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded-r-md hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </div>
  );
}
