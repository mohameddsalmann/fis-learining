document.addEventListener('DOMContentLoaded', function() {
    const button = document.getElementById('actionButton');
    const responseContainer = document.getElementById('responseContainer');

    button.addEventListener('click', function() {
        // Disable button during processing
        button.disabled = true;
        button.textContent = 'Processing...';

        // Simulate API call delay (optional, remove if you want instant response)
        setTimeout(() => {
            // Return the response
            const response = {
                success: true,
                next_step: 'pending_approval'
            };

            // Display the response
            displayResponse(response);

            // Re-enable button
            button.disabled = false;
            button.textContent = 'Click Me';
        }, 500);
    });

    function displayResponse(response) {
        responseContainer.innerHTML = `
            <div class="response-content">
                <div class="success-badge">✓ Success</div>
                <div style="margin-top: 15px;">
                    <strong>Response:</strong>
                    <pre>${JSON.stringify(response, null, 2)}</pre>
                </div>
                <div class="status-badge">Next Step: ${response.next_step}</div>
            </div>
        `;
        responseContainer.classList.add('show');
    }
});

