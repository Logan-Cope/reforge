// Read-only demo (DEMO_READ_ONLY=1). The server refuses every write with 403; this
// script only keeps the UI honest: write controls render disabled, and htmx never
// sends a non-GET request. GET polling and lazy-loaded partials keep working.
(function () {
  function lock(root) {
    root.querySelectorAll("[hx-post],[hx-put],[hx-patch],[hx-delete]").forEach(function (el) {
      var controls = el.matches("form") ? el.querySelectorAll("button, input, textarea, select") : [el];
      controls.forEach(function (c) { c.disabled = true; });
      el.setAttribute("aria-disabled", "true");
    });
  }
  document.addEventListener("DOMContentLoaded", function () { lock(document); });
  document.body.addEventListener("htmx:afterSwap", function (e) { lock(e.target); });
  document.body.addEventListener("htmx:beforeRequest", function (e) {
    var verb = (e.detail.requestConfig && e.detail.requestConfig.verb) || "get";
    if (verb.toLowerCase() !== "get") e.preventDefault();
  });
})();
