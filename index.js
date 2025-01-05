let mediaRecorder;
let audioChunks = [];

// Function to send audio chunk to backend for real-time transcription
const sendAudioChunk = async (chunk) => {
    const formData = new FormData();
    formData.append("file", chunk, "chunk.wav");

    try {
        const response = await fetch("http://127.0.0.1:8000/transcribe", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById("transcription").value = data.transcription;
        } else {
            console.error("Error in transcription.");
        }
    } catch (error) {
        console.error("Error sending audio chunk:", error);
    }
};

document.getElementById("startRecording").addEventListener("click", async () => {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
        // Send each chunk to the backend for transcription
        sendAudioChunk(event.data);
    };

    mediaRecorder.onstop = () => {
        document.getElementById("startRecording").disabled = false;
        document.getElementById("stopRecording").disabled = true;
    };

    mediaRecorder.start();
    document.getElementById("startRecording").disabled = true;
    document.getElementById("stopRecording").disabled = false;
});

document.getElementById("stopRecording").addEventListener("click", () => {
    mediaRecorder.stop();
    document.getElementById("startRecording").disabled = false;
    document.getElementById("stopRecording").disabled = true;
});
