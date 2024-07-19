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

    const result = await response.json();
    document.getElementById('results').innerHTML = result.message.replace(/\n/g, '<br>');
});
