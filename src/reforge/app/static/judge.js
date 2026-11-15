/* Judgment Pass: watch-first review glue (vanilla JS, no framework).
   - Syncs the sticky reel player with the beat rail (live chip + auto-scroll).
   - Chip tap seeks the player and jumps the page to that beat's card.
   - Beat frames lazily load a looping proxy of the SOURCE moment on first tap.
   - "In reel" / frame tap seeks the main player to the beat start (one video, no dupes). */
(function () {
  "use strict";

  var player = document.getElementById("reel-player");
  var rail = document.getElementById("beat-rail");
  var VERDICT_CLASSES = ["beat-chip-open", "beat-chip-keep", "beat-chip-weak",
                         "beat-chip-swapped", "beat-chip-flagged"];

  if (player) document.body.classList.add("has-player");

  function chips() {
    return rail ? Array.prototype.slice.call(rail.querySelectorAll(".beat-chip")) : [];
  }

  // --- player -> rail: highlight + follow the live beat -----------------------
  var liveChip = null;
  if (player && rail) {
    player.addEventListener("timeupdate", function () {
      var t = player.currentTime;
      var next = null;
      chips().forEach(function (c) {
        var on = t >= parseFloat(c.dataset.start) && t < parseFloat(c.dataset.end);
        c.classList.toggle("beat-live", on);
        if (on) next = c;
      });
      if (next && next !== liveChip) {
        next.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      }
      liveChip = next;
    });
    player.addEventListener("play", pauseAllProxies);

    // rail -> player + page: tap a chip, watch that beat, land on its card
    rail.addEventListener("click", function (e) {
      var chip = e.target.closest(".beat-chip");
      if (!chip) return;
      seekReel(parseFloat(chip.dataset.start));
      var card = document.getElementById("slot-" + chip.dataset.slot);
      if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  // compact the sticky dock once the review scrolls into the beats
  var dock = document.querySelector(".player-dock");
  if (dock) {
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        dock.classList.toggle("dock-mini", window.scrollY > 120);
        ticking = false;
      });
    }, { passive: true });
  }

  function seekReel(start) {
    if (!player || isNaN(start)) return;
    player.currentTime = start;
    player.play();
  }

  function pauseAllProxies() {
    document.querySelectorAll("video.beat-src").forEach(function (v) {
      if (!v.paused) v.pause();
      var wrap = v.parentElement;
      if (wrap) wrap.classList.remove("is-playing");
    });
  }

  // --- lazy source proxies: load on first tap, loop the [in, in+dur] window ---
  function toggleProxy(overlay) {
    var wrap = overlay.parentElement;
    var vid = wrap && wrap.querySelector("video.beat-src");
    if (!vid) return;
    if (!vid.dataset.armed) {
      vid.dataset.armed = "1";
      var start = parseFloat(vid.dataset.in) || 0;
      var dur = parseFloat(vid.dataset.dur) || 0;
      // Media fragment = native start hint; belt-and-braces seeks below because a missed
      // seek here meant swapping on a preview of the WRONG seconds (the setup, not the moment).
      vid.src = vid.dataset.src + "#t=" + start;
      if (vid.readyState >= 1) {
        vid.currentTime = start; // metadata already in (cache) - seek now, no event race
      } else {
        vid.addEventListener("loadedmetadata", function () { vid.currentTime = start; },
                             { once: true });
      }
      vid.addEventListener("playing", function () {
        if (vid.currentTime < start - 0.5) vid.currentTime = start; // fragment ignored: nudge
      });
      vid.addEventListener("timeupdate", function () {
        if (dur > 0 && vid.currentTime > start + dur) vid.currentTime = start; // window loop
        else if (vid.currentTime < start - 0.5) vid.currentTime = start; // drifted before in-point
      });
    }
    if (vid.paused) {
      pauseAllProxies();
      if (player && !player.paused) player.pause(); // one thing plays at a time
      vid.play();
      wrap.classList.add("is-playing");
    } else {
      vid.pause();
      wrap.classList.remove("is-playing");
    }
  }

  // Capture phase so a proxy tap never bubbles into the swap button / frame seek.
  document.addEventListener("click", function (e) {
    var overlay = e.target.closest(".play-overlay");
    if (!overlay) return;
    e.stopPropagation();
    e.preventDefault();
    toggleProxy(overlay);
  }, true);

  // --- "other" flag requires a typed note: block the post, nudge the field ------
  // Capture phase so the block runs before HTMX's own click handler on the chip.
  document.addEventListener("click", function (e) {
    var chip = e.target.closest("[data-requires-note]");
    if (!chip) return;
    var form = chip.closest("form");
    var note = form && form.querySelector('input[name="note"]');
    if (note && note.value.trim()) return; // note present: let HTMX submit it
    e.stopPropagation();
    e.preventDefault();
    if (note) {
      note.placeholder = "Type what's wrong first";
      note.classList.add("field-shake");
      note.addEventListener("animationend", function () {
        note.classList.remove("field-shake");
      }, { once: true });
      note.focus();
    }
  }, true);

  // --- card -> player: "In reel" chip or the frame itself seeks the main reel --
  document.addEventListener("click", function (e) {
    if (!player) return;
    var hit = e.target.closest("[data-seek], [data-beat-frame]");
    if (!hit) return;
    var card = hit.closest(".beat-card");
    if (!card) return;
    seekReel(parseFloat(card.dataset.start));
    if (dock) dock.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // --- re-render finished: load the fresh cut into the main player -------------
  // The status banner's done-state button carries data-new-src (the reel's updated
  // output). One tap swaps the player source and plays from the top; a full page
  // load later restores the normal re-render button state.
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-load-new-cut]");
    if (!btn) return;
    var newSrc = btn.dataset.newSrc;
    if (player && newSrc) {
      pauseAllProxies();
      player.src = newSrc;
      player.currentTime = 0;
      player.play();
      if (dock) dock.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    btn.textContent = "Playing the new cut";
    btn.disabled = true;
  });

  // --- keep rail tints honest after HTMX swaps a slot card ---------------------
  function syncRail() {
    chips().forEach(function (chip) {
      var card = document.querySelector('.beat-card[data-slot="' + chip.dataset.slot + '"]');
      if (!card) return;
      VERDICT_CLASSES.forEach(function (cls) { chip.classList.remove(cls); });
      chip.classList.add("beat-chip-" + (card.dataset.verdict || "open"));
    });
  }
  document.body.addEventListener("htmx:afterSwap", syncRail);
})();
