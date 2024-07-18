document.getElementById('queryForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const queryText = document.getElementById('queryText').value;
    const response = await fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query_text: queryText })
    });

    try {
        const result = await response.json();
        console.log(result); // Check the response in browser console

        if (result.error) {
            document.getElementById('results').innerHTML = `<p>Error: ${result.error}</p>`;
        } else if (result.message) {
            document.getElementById('results').innerHTML = `<p>${result.message}</p>`;
        } else if (result.length > 0) {
            let html = '<ul>';
            result.forEach(item => {
                html += `<li>URL: <a href="${item.url}" target="_blank">${item.url}</a></li>`;
                html += `<li>Text: ${item.text}</li>`;
                html += `<li>Similarity: ${item.similarity}</li><br>`;
            });
            html += '</ul>';
            document.getElementById('results').innerHTML = html;
        } else {
            document.getElementById('results').innerHTML = `<p>No results found.</p>`;
        }
    } catch (error) {
        console.error('Error parsing JSON response:', error);
        document.getElementById('results').innerHTML = `<p>Error: ${error.message}</p>`;
    }
});
