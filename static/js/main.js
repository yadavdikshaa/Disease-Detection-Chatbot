$(document).ready(function () {
  symptoms = JSON.parse(symptoms);
  let input = $("#message-text");
  let sendBtn = $("#send");
  let startOverBtn = $("#start-over");
  let dataList = $("#symptoms-list");
  let chat = $("#conversation");

  // Handler for any input on the message input field
  input.on("input", function () {
    let insertedValue = $(this).val();
    $("#symptoms-list ul").remove();

    if (insertedValue.length > 1) {
      let suggestedSymptoms = $.fn.getSuggestedSymptoms(insertedValue);
      if (suggestedSymptoms.length === 0) {
        dataList.removeClass("show");
      } else {
        let ul = $("<ul></ul>");
        for (let i = 0; i < suggestedSymptoms.length; i++) {
          let li = $("<li></li>").text(suggestedSymptoms[i]);
          ul.append(li);
        }
        dataList.append(ul);
        dataList.addClass("show");
      }
    } else {
      dataList.removeClass("show");
    }
  });

  startOverBtn.on("click", function () {
    $.fn.startOver();
  });

  sendBtn.on("click", function () {
    $.fn.handleUserMessage();
  });

  // Handler for click on one of the suggested symptoms
  dataList.on("click", "li", function () {
    input.val($(this).text());
    dataList.removeClass("show");
    input.focus();
  });

  input.on("blur", function () {
    setTimeout(() => {
      dataList.removeClass("show");
    }, 200);
  });

  input.on("keypress", function (e) {
    if (e.which == 13) {
      e.preventDefault();
      $.fn.handleUserMessage();
    }
  });

  // Handler function for sending a message
  $.fn.handleUserMessage = function () {
    if (input.val().trim()) {
      $.fn.appendUserMessage();
      $.fn.showTypingIndicator();
      $.fn.getPredictedSymptom();
      input.val("");
      dataList.removeClass("show");
      setTimeout(() => {
        chat.scrollTop(chat[0].scrollHeight);
      }, 100);
    }
  };

  $.fn.startOver = function () {
    $.fn.getPredictedSymptom(true);
    $("#conversation").empty();
    const text =
      "Welcome! I'm <strong>Meddy</strong>, your medical assistant. I'm here to help you understand your symptoms better.<br><br>Please describe the symptoms you're experiencing. When you're done, type <strong>'Done'</strong> to get a diagnosis.<br><br><em>Note: Enter as many symptoms as possible for the most accurate prediction.</em>";
    $.fn.appendBotMessage(text);
    input.val("");
    dataList.removeClass("show");
    chat.scrollTop(0);
  };

  // Creates the newly sent message element
  $.fn.appendUserMessage = function () {
    var text = input.val().trim();
    if (!text) return;
    
    let messageHtml = `
      <div class="message-wrapper user-message">
        <div class="message-avatar">
          <i class="fas fa-user"></i>
        </div>
        <div class="message-content">
          <div class="message-bubble user-bubble">
            <div class="message-text">${$.fn.escapeHtml(text)}</div>
          </div>
          <div class="message-time">Just now</div>
        </div>
      </div>
    `;
    $("#conversation").append(messageHtml);
    chat.scrollTop(chat[0].scrollHeight);
  };

  $.fn.appendBotMessage = function (text) {
    let messageHtml = `
      <div class="message-wrapper bot-message">
        <div class="message-avatar">
          <i class="fas fa-user-doctor"></i>
        </div>
        <div class="message-content">
          <div class="message-bubble bot-bubble">
            <div class="message-text">${text}</div>
          </div>
          <div class="message-time">Just now</div>
        </div>
      </div>
    `;
    $("#conversation").append(messageHtml);
    chat.scrollTop(chat[0].scrollHeight);
  };

  $.fn.showTypingIndicator = function () {
    let typingHtml = `
      <div class="message-wrapper bot-message typing-indicator-wrapper">
        <div class="message-avatar">
          <i class="fas fa-user-doctor"></i>
        </div>
        <div class="message-content">
          <div class="message-bubble bot-bubble typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
          </div>
        </div>
      </div>
    `;
    $("#conversation").append(typingHtml);
    chat.scrollTop(chat[0].scrollHeight);
  };

  $.fn.removeTypingIndicator = function () {
    $(".typing-indicator-wrapper").remove();
  };

  // Retreives prediction to show as bot message
  $.fn.getPredictedSymptom = function (again) {
    var text = input.val();
    if (again) text = "done";

    $.ajax({
      url: "/symptom",
      data: JSON.stringify({ sentence: text }),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      type: "POST",
      success: function (response) {
        $.fn.removeTypingIndicator();
        if (!again) {
          $.fn.appendBotMessage(response);
        } else {
          const text =
            "Welcome! I'm <strong>Meddy</strong>, your medical assistant. I'm here to help you understand your symptoms better.<br><br>Please describe the symptoms you're experiencing. When you're done, type <strong>'Done'</strong> to get a diagnosis.<br><br><em>Note: Enter as many symptoms as possible for the most accurate prediction.</em>";
          $.fn.appendBotMessage(text);
        }
        chat.scrollTop(chat[0].scrollHeight);
      },
      error: function (xhr, status, error) {
        $.fn.removeTypingIndicator();
        $.fn.appendBotMessage("Sorry, I encountered an error. Please try again.");
        console.error("Error:", error);
      },
    });
  };

  $.fn.getSuggestedSymptoms = function (val) {
    let suggestedSymptoms = [];
    val = val.toLowerCase();
    $.each(symptoms, function (i, v) {
      if (v.toLowerCase().includes(val)) {
        suggestedSymptoms.push(v);
      }
    });
    return suggestedSymptoms.slice(0, 5);
  };

  $.fn.escapeHtml = function (text) {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return text.replace(/[&<>"']/g, function (m) {
      return map[m];
    });
  };
});
