document.addEventListener('DOMContentLoaded', function() {
    const urlForm = document.getElementById('urlForm');
    const urlInput = document.getElementById('urlInput');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const productsList = document.getElementById('productsList');
    const urlDisplay = document.getElementById('urlDisplay');
    const errorAlert = document.getElementById('errorAlert');
    const copyButton = document.getElementById('copyButton');
    const newSearchButton = document.getElementById('newSearchButton');

    urlForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Сброс предыдущих результатов
        errorAlert.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        
        // Показать загрузку
        loadingSpinner.classList.remove('d-none');
        
        try {
            const response = await fetch('/extract-products/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: urlInput.value }),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Произошла ошибка при анализе страницы');
            }
            
            const data = await response.json();
            
            // Отобразить результаты
            urlDisplay.textContent = `Анализирован URL: ${data.url}`;
            productsList.innerHTML = '';
            
            if (data.products && data.products.length > 0) {
                data.products.forEach(product => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = product;
                    productsList.appendChild(li);
                });
            } else {
                productsList.innerHTML = '<li class="list-group-item text-muted">Мебель не найдена</li>';
            }
            
            resultsContainer.classList.remove('d-none');
        } catch (error) {
            errorAlert.textContent = error.message;
            errorAlert.classList.remove('d-none');
        } finally {
            loadingSpinner.classList.add('d-none');
        }
    });
    
    copyButton.addEventListener('click', function() {
        const products = Array.from(productsList.children)
            .map(li => li.textContent)
            .join('\n');
        
        navigator.clipboard.writeText(products).then(() => {
            Swal.fire({
                icon: 'success',
                title: 'Скопировано!',
                text: 'Список мебели скопирован в буфер обмена',
                timer: 2000,
                showConfirmButton: false
            });
        });
    });
    
    newSearchButton.addEventListener('click', function() {
        urlForm.reset();
        resultsContainer.classList.add('d-none');
        errorAlert.classList.add('d-none');
        urlInput.focus();
    });
});