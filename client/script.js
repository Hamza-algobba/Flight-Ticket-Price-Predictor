function sendData() {
  var formData = {
    airline: document.getElementById("airline").value,
    source_city: document.getElementById("source_city").value,
    destination_city: document.getElementById("destination_city").value,
    departure_time: document.getElementById("departure_time").value,
    arrival_time: document.getElementById("arrival_time").value,
    stops: document.getElementById("stops").value,
    class: document.getElementById("flight_class").value,
    days_left: parseInt(document.getElementById("days_left").value),
    duration: parseInt(document.getElementById("duration").value),
  };
  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.prediction) {
        var roundedPrediction = Math.round(data.prediction);
        document.getElementById("result").innerText =
          "Predicted Price: " + roundedPrediction;
      } else {
        document.getElementById("result").innerText = "Error: " + data.error;
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      document.getElementById("result").innerText = "Error: " + error.message;
    });
}
