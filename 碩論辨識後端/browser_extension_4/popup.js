let videoStream = null;

document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.createElement('video');
    videoElement.style.display = 'none';
    document.body.appendChild(videoElement);

    // Start video stream and capture frames
    function startVideoStream() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoStream = stream;
                videoElement.srcObject = stream;
                videoElement.play();
                captureFrame(); // Start capturing frames
            })
            .catch(error => console.error('Error accessing media devices.', error));
    }

    function captureFrame() {
        if (!videoStream) return;

        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth || 640; // Fallback width
        canvas.height = videoElement.videoHeight || 480; // Fallback height
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            // Send the frame to the backend
            fetch('http://120.107.172.140:5000/process-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Pose Data:', data);
                chrome.storage.local.set({ poseData: data });
            })
            .catch(error => console.error('Error sending frame:', error));
        }, 'image/jpeg');

        // Continue capturing frames every second
        setTimeout(captureFrame, 1000);
    }

    // Add event listeners to buttons
    document.getElementById('startBtn').addEventListener('click', startVideoStream);
    document.getElementById('stopBtn').addEventListener('click', () => {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            console.log("Video stream stopped.");
        }
    });
});
