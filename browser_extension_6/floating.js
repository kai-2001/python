document.addEventListener('DOMContentLoaded', () => {
    window.resizeTo(338, 538);
    const videoElement = document.getElementById('videoElement');
    const poseLabelElement = document.getElementById('poseLabel');
    const poseConfidenceElement = document.getElementById('poseConfidence');
    const war = document.getElementById('warning');
    const inputNumberElement = document.getElementById('inputNumber');
    const distanceElement = document.getElementById('distance');
    
    let nonULabelDuration = 0;
    const checkInterval = 200; // 每秒檢查10次
    const requiredDuration = 10000; // 10秒
    let audio = null; // 音效對象的引用

    // 播放音效的函數
    function playAudio() {
        if (!audio) {
            audio = new Audio('7403.mp3'); // 替換成音效檔案的路徑
            audio.loop = true; // 設定為循環播放（如果需要）
            audio.play();
        }
    }

    // 停止音效的函數
    function stopAudio() {
        if (audio) {
            audio.pause();
            audio.currentTime = 0; // 重置音效
            audio = null; // 清除引用，準備下次播放
        }
    }

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

        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
            // 如果尚未有寬高，延遲再試
            setTimeout(captureFrame, checkInterval);
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            const thresholdValue = inputNumberElement.value;
            formData.append('threshold', thresholdValue);

            fetch('http://120.107.172.140:5000/process-image', {
                method: 'POST',
                body: formData
            })
            .then(async response => {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    const data = await response.json();
                    console.log('Pose Data:', data);
                    chrome.runtime.sendMessage({ action: 'updatePoseData', data: data });
                } else {
                    const text = await response.text(); // 把 HTML 或錯誤訊息印出來
                    console.error("❌ Server did not return JSON. Raw response:\n", text);
                }
            })
            .catch(error => console.error('Error sending frame:', error));
        }, 'image/jpeg');

        setTimeout(captureFrame, checkInterval);
    }

    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'updatePoseData') {
            const data = message.data || { label: 'N/A', confidence: 0.0 };
            poseLabelElement.textContent = data.label || 'N/A';
            poseConfidenceElement.textContent = data.confidence ? data.confidence.toFixed(2) : 'N/A';
            distanceElement.textContent = data.distance ? data.distance.toFixed(2) : 'N/A';


            if (data.label !== 'u') {
                nonULabelDuration += checkInterval;
                if (nonULabelDuration >= requiredDuration) {
                    playAudio();
                    if(data.distance<40)
                        war.textContent = "Please maintain good posture and screen distance.";
                    else
                        war.textContent = "Please sit up straight.";
                }
                else
                {
                    if(data.distance<40)
                    {
                        war.textContent = "Keep a proper distance from the screen.";
                        playAudio();
                    }
                    else
                    {
                        stopAudio(); // 停止音效
                        war.textContent = "";
                        
                    }
                }
            }
            else{
                nonULabelDuration = 0;
                if(data.distance<40)
                {
                    war.textContent = "Keep a proper distance from the screen.";
                    playAudio();
                }
                else
                {
                    stopAudio(); // 停止音效
                    war.textContent = "";
                    
                }
                
            }
        }
    });
    

    startVideoStream();
});

