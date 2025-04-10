
document.addEventListener('DOMContentLoaded', function() {
    // ��¥ �⺻�� ����
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today.toISOString().split('T')[0];
    
    // ������ ��������
    document.getElementById('fetchData').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;
        
        fetch('/api/fetch_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                start_date: startDate,
                end_date: endDate
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // ��ú��� ǥ��
                const dashboardPath = data.data.dashboard_path;
                document.getElementById('dashboardFrame').src = dashboardPath;
                
                alert('Data fetched successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching data.');
        });
    });
    
    // �� �н�
    document.getElementById('trainModels').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        
        fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // �� ��� ǥ��
                const evaluationPath = data.evaluation_path;
                document.getElementById('dashboardFrame').src = evaluationPath;
                
                alert('Models trained successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while training models.');
        });
    });
    
    // ����
    document.getElementById('predict').addEventListener('click', function() {
        const symbol = document.getElementById('symbol').value;
        const modelType = document.getElementById('modelType').value;
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // ���� ��� ǥ��
                const visualizationPath = data.prediction_path;
                document.getElementById('dashboardFrame').src = visualizationPath;
                
                alert('Prediction completed successfully!');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making predictions.');
        });
    });
});
        