document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('videoElement');
    const poseLabelElement = document.getElementById('poseLabel');
    const poseConfidenceElement = document.getElementById('poseConfidence');

    function startVideoStream() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoElement.srcObject = stream;
                videoElement.play();
                captureFrame();
            })
            .catch(error => console.error('Error accessing media devices.', error));
    }

    function captureFrame() {
        if (!videoElement.srcObject) return;

        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        // 將 canvas 轉換為 blob 並發送到後端
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            fetch('http://120.107.172.140:5000/process-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Pose Data:', data);
                chrome.runtime.sendMessage({ action: 'updatePoseData', data: data });
            })
            .catch(error => console.error('Error sending frame:', error));
        }, 'image/jpeg');

        // 設置捕獲間隔
        setTimeout(captureFrame, 1000);
    }

    // 設置消息接收處理
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'updatePoseData') {
            const data = message.data || { label: 'N/A', confidence: 0.0 };
            poseLabelElement.textContent = data.label;
            poseConfidenceElement.textContent = data.confidence.toFixed(2);
        }
    });

    startVideoStream();
});
