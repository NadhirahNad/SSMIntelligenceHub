import React from "react";
import MessageBubble from "./MessageBubble";

export default function ChatBox({ messages }) {
  return (
    <div className="bg-white shadow-md rounded-lg w-[600px] h-[500px] overflow-y-auto p-4">
      {messages.map((msg, index) => (
        <MessageBubble key={index} sender={msg.sender} text={msg.text} />
      ))}
    </div>
  );
}
