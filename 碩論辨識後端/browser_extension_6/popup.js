document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    function startVideoStream() {
        chrome.runtime.sendMessage({ action: 'start' });
    }

    function stopVideoStream() {
        chrome.runtime.sendMessage({ action: 'stop' });
    }

    startBtn.addEventListener('click', startVideoStream);
    stopBtn.addEventListener('click', stopVideoStream);
});
