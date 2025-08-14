let sessionId = null;
let fetchInterval = null;

function startDetection() {
    fetch('http://120.107.172.140:5000/start-detection', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            sessionId = data.session_id; // Save the session ID
            console.log("Detection started:", data.message);
            // Start fetching pose data periodically
            fetchPoseData();
        })
        .catch(error => console.error('Error starting detection:', error));
}

function stopDetection() {
    if (sessionId) {
        fetch('http://120.107.172.140:5000/stop-detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        })
            .then(response => response.json())
            .then(data => {
                console.log("Detection stopped:", data.message);
                sessionId = null; // Clear the session ID
                clearInterval(fetchInterval); // Stop fetching pose data
            })
            .catch(error => console.error('Error stopping detection:', error));
    }
}

function fetchPoseData() {
    if (sessionId) {
        fetch(`http://120.107.172.140:5000/get-pose?session_id=${sessionId}`)
            .then(response => response.json())
            .then(data => {
                console.log("Pose data fetched:", data);  // Print to verify data
                chrome.storage.local.set({ poseData: data });  // Save pose data to local storage
            })
            .catch(error => console.error('Error fetching pose data:', error));
    }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'startDetection') {
        startDetection();
        sendResponse({ message: 'Detection started' });
    } else if (request.action === 'stopDetection') {
        stopDetection();
        sendResponse({ message: 'Detection stopped' });
    }
});

// Periodically fetch pose data while detection is running
fetchInterval = setInterval(fetchPoseData, 1000);  // Fetch pose data every 1 second (adjust as needed)
