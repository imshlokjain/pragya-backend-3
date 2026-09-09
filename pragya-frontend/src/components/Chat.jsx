import { useState } from "react";


export default function Chat({
  zoneId,
  zoneName,
  onSend,
}) {

  const [message, setMessage] =
    useState("");


  const [messages, setMessages] =
    useState([
      {
        role: "assistant",

        content:
          `I'm PRAGYA Intelligence. I'm currently analysing ${zoneName}. Ask me about flood risk, emergency actions, or current conditions.`,
      },
    ]);


  const [loading, setLoading] =
    useState(false);



  async function handleSubmit(e) {

    e.preventDefault();


    if (!message.trim() || loading) {
      return;
    }


    if (!zoneId) {

      setMessages((prev) => [
        ...prev,

        {
          role: "assistant",

          content:
            "Please select a monitoring region before asking a question.",
        },

      ]);

      return;
    }


    const userMessage =
      message.trim();


    setMessages((prev) => [
      ...prev,

      {
        role: "user",
        content: userMessage,
      },

    ]);


    setMessage("");
    setLoading(true);


    try {

      const response =
        await onSend(
          userMessage,
          zoneId
        );


      setMessages((prev) => [
        ...prev,

        {
          role: "assistant",

          content:
            response?.answer ||
            "I couldn't generate a response.",
        },

      ]);


    } catch (err) {

      setMessages((prev) => [
        ...prev,

        {
          role: "assistant",

          content:
            `Unable to contact PRAGYA Intelligence: ${err.message}`,
        },

      ]);


    } finally {

      setLoading(false);

    }

  }



  return (

    <div className="panel chat-panel">


      <div className="panel-header">

        <div>

          <p className="eyebrow">
            AI ASSISTANT
          </p>


          <h3>
            PRAGYA Intelligence
          </h3>


          <small className="chat-zone">
            📍 {zoneName || "No region selected"}
          </small>

        </div>


        <div className="ai-status">

          <span></span>

          Online

        </div>

      </div>



      <div className="chat-messages">

        {messages.map(
          (item, index) => (

            <div
              key={index}
              className={`message ${item.role}`}
            >

              <div className="message-avatar">

                {item.role === "assistant"
                  ? "P"
                  : "U"}

              </div>


              <div className="message-content">

                {item.content}

              </div>

            </div>

          )
        )}



        {loading && (

          <div className="message assistant">

            <div className="message-avatar">
              P
            </div>


            <div className="message-content typing">

              <span></span>
              <span></span>
              <span></span>

            </div>

          </div>

        )}

      </div>



      <form
        className="chat-input"
        onSubmit={handleSubmit}
      >

        <input
          value={message}

          onChange={(e) =>
            setMessage(e.target.value)
          }

          placeholder={
            `Ask about ${zoneName || "flood conditions"}...`
          }

        />


        <button
          type="submit"
          disabled={loading || !zoneId}
        >
          ↑
        </button>

      </form>

    </div>

  );
}
