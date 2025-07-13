// document.addEventListener('DOMContentLoaded', () => {
//     const uploadBtn = document.getElementById('uploadBtn');
//     const videoFile = document.getElementById('videoFile');
//     const progressContainer = document.getElementById('progressContainer');
//     const progressBar = document.getElementById('progressBar');

//     uploadBtn.addEventListener('click', async () => {
//         if (!videoFile.files[0]) {
//             alert('Please select a video file');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('video', videoFile.files[0]);

//         progressContainer.classList.remove('hidden');
//         progressBar.style.width = '0%';

//         try {
//             const response = await fetch('/upload', {
//                 method: 'POST',
//                 body: formData
//             });
//             const data = await response.json();
//             if (!response.ok) {
//                 throw new Error(data.error || 'Upload failed');
//             }

//             // Poll for progress
//             const pollProgress = async () => {
//                 try {
//                     const response = await fetch('/progress');
//                     const progressData = await response.json();
//                     if (progressData.error) {
//                         throw new Error(progressData.error);
//                     }
//                     progressBar.style.width = `${progressData.progress}%`;
//                     if (progressData.done) {
//                         if (data.redirect) {
//                             window.location.href = data.redirect;
//                         }
//                     } else {
//                         setTimeout(pollProgress, 1000); // Poll every second
//                     }
//                 } catch (error) {
//                     alert('Error during processing: ' + error.message);
//                     progressContainer.classList.add('hidden');
//                 }
//             };
//             pollProgress();
//         } catch (error) {
//             alert('Error uploading video: ' + error.message);
//             progressContainer.classList.add('hidden');
//         }
//     });
// });

// document.getElementById('uploadForm').addEventListener('submit', function(e) {
//     e.preventDefault();
    
//     const formData = new FormData(this);
    
//     fetch('/upload', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.redirect) {
//             window.location.href = data.redirect;
//         } else if (data.error) {
//             alert(data.error);
//         }
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });


document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('uploadBtn');
    const videoFile = document.getElementById('videoFile');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');

    uploadBtn.addEventListener('click', async () => {
        if (!videoFile.files[0]) {
            alert('Please select a video file');
            return;
        }

        uploadBtn.disabled = true; // Disable button to prevent multiple submissions
        uploadBtn.textContent = 'Processing...';

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            console.log('Upload response:', data);

            // Poll for progress
            const pollProgress = async () => {
                try {
                    const response = await fetch('/progress');
                    const progressData = await response.json();
                    console.log('Progress data:', progressData);
                    if (progressData.error) {
                        throw new Error(progressData.error);
                    }
                    progressBar.style.width = `${progressData.progress}%`;
                    if (progressData.done) {
                        console.log('Processing complete, redirecting to:', data.redirect);
                        window.location.href = data.redirect;
                    } else {
                        setTimeout(pollProgress, 1000); // Poll every second
                    }
                } catch (error) {
                    console.error('Polling error:', error.message);
                    alert('Error during processing: ' + error.message);
                    progressContainer.classList.add('hidden');
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = 'Upload & Analyze';
                }
            };
            pollProgress();
        } catch (error) {
            console.error('Upload error:', error.message);
            alert('Error uploading video: ' + error.message);
            progressContainer.classList.add('hidden');
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload & Analyze';
        }
    });
});


