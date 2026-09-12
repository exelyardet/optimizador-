# Optimizador de Carteras

Aplicación Streamlit para optimización de portfolios usando teoría de Markowitz
(frontera eficiente, portfolio de Sharpe óptimo, mínima volatilidad, correlación,
rendimientos, Value at Risk y stress testing).

Basado en el proyecto original de [xfrancomaciel/optimizador-carteras](https://github.com/xfrancomaciel/optimizador-carteras),
con una modificación en el panel lateral: además de definir un **peso mínimo
igual para todos los activos**, ahora se puede elegir **personalizar el peso
mínimo activo por activo**.

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Pensado para desplegarse en [Streamlit Community Cloud](https://share.streamlit.io/),
apuntando a `app.py` en la rama `main`.
