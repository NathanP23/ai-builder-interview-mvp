const defaultPrompt =
  "Customer 1842 says order O-991 was charged twice. Check the customer, order, transactions, and refund policy, then prepare a refund if policy allows.";

function App() {
  const [message, setMessage] = React.useState(defaultPrompt);
  const [status, setStatus] = React.useState("Ready");
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState("");

  async function sendMessage(event) {
    event.preventDefault();
    setStatus("Sending request to FastAPI...");
    setError("");
    setResult(null);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setResult(await response.json());
      setStatus("Agent finished.");
    } catch (err) {
      setError(err.message);
      setStatus("Request failed.");
    }
  }

  return React.createElement(
    "section",
    { className: "shell" },
    React.createElement("header", null, [
      React.createElement("p", { className: "eyebrow", key: "eyebrow" }, "Checkpoint 15"),
      React.createElement("h1", { key: "title" }, "Enterprise Support Agent"),
      React.createElement(
        "p",
        { className: "subtitle", key: "subtitle" },
        "Browser -> FastAPI -> LangGraph -> tools/RAG -> JSON response"
      ),
    ]),
    React.createElement("form", { onSubmit: sendMessage, className: "composer" }, [
      React.createElement("label", { htmlFor: "message", key: "label" }, "Support request"),
      React.createElement("textarea", {
        id: "message",
        key: "textarea",
        value: message,
        onChange: (event) => setMessage(event.target.value),
      }),
      React.createElement("button", { key: "button", type: "submit" }, "Send to agent"),
    ]),
    React.createElement("p", { className: "status" }, status),
    error && React.createElement("pre", { className: "error" }, error),
    result &&
      React.createElement("section", { className: "result" }, [
        React.createElement("h2", { key: "answer-title" }, "Answer"),
        React.createElement("p", { key: "answer" }, result.answer),
        React.createElement("h2", { key: "details-title" }, "Structured result"),
        React.createElement("pre", { key: "details" }, JSON.stringify(result, null, 2)),
      ])
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
