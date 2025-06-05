import pandas as pd
from datetime import datetime

class RFMAnalyzer:
    def calculate_rfm(self, data):
        # Определение текущей даты для расчета Recency
        now = datetime.now()
        
        # Группировка по CustomerID
        rfm = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (now - x.max()).days,
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        
        return rfm