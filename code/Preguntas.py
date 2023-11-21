import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm  
import seaborn as sns
tqdm.pandas()

# Cargar el conjunto de datos desde el archivo CSV
df = pd.read_csv('DE_videos.csv')

# A. ¿Qué categorías de videos son las de mayor tendencia?

# Convertir la columna 'publish_time' a tipo datetime y ajustar zona horaria
df['publish_time'] = pd.to_datetime(df['publish_time'], utc=True).dt.tz_convert(None)

# Calcular las visitas totales por categoría
visitas_por_categoria = df.groupby('category_id')['views'].sum()

# Calcular la cantidad de videos por categoría
cantidad_videos_por_categoria = df.groupby('category_id').size()

# Calcular el promedio de visitas por categoría
promedio_visitas_por_categoria = visitas_por_categoria / cantidad_videos_por_categoria

# Calcular la diferencia de días entre 'publish_time' y '1/11/2023'
df['diferencia_dias'] = (pd.to_datetime('2023-11-01') - df['publish_time']).dt.days

# Calcular la diferencia promedio de días por categoría
promedio_diferencia_dias_por_categoria = df.groupby('category_id')['diferencia_dias'].mean()

# Calcular el promedio de visitas diarias por categoría
promedio_visitas_diarias = promedio_visitas_por_categoria / promedio_diferencia_dias_por_categoria

# Crear un DataFrame con los resultados
resultados = pd.DataFrame({
    'Categoria': promedio_visitas_diarias.index,
    'Promedio_Visitas_Diarias': promedio_visitas_diarias.values
})

# Graficar los resultados utilizando seaborn para una visualización atractiva
plt.figure(figsize=(10, 6))
sns.barplot(x='Categoria', y='Promedio_Visitas_Diarias', data=resultados, palette='viridis')
plt.xlabel('Categoría')
plt.ylabel('Visitas Diarias Promedio')
plt.title('Promedio de Visitas Diarias por Categoría')
plt.xticks(rotation=90)  # Rotar etiquetas para mejor visualización
plt.tight_layout()
plt.show()

# B. ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?

# Calcular el promedio de likes por categoría
promedio_likes_por_categoria = df.groupby('category_id')['likes'].mean().sort_values(ascending=False)

# Crear un gráfico de barras horizontal con seaborn para visualizar los promedios de likes
plt.figure(figsize=(10, 6))
plot = sns.barplot(x=promedio_likes_por_categoria.values, y=promedio_likes_por_categoria.index, palette='coolwarm')
plt.xlabel('Promedio de Likes')
plt.ylabel('Categoría')
plt.title('Promedio de Likes por Categoría')
plot.set_xticklabels(plot.get_xticklabels(), rotation=45)  # Rotar etiquetas
plt.tight_layout()
plt.show()

# Imprimir los resultados
print(promedio_likes_por_categoria)


# C. ¿Qué categorías de videos tienen la mejor proporción (ratio) de "Me gusta" / "No me gusta"?

# Calcular el ratio de likes por visita y filtrar para evitar divisiones por cero
df['likes_per_view'] = df['likes'] / df['views'].where(df['views'] > 0, other=1)

# Calcular el promedio del ratio de likes por categoría
ratio_likes_por_categoria = df.groupby('category_id')['likes_per_view'].mean().sort_values(ascending=False)

# Crear un gráfico de barras horizontal con seaborn para visualizar los ratios de likes por visita
plt.figure(figsize=(10, 6))
plot = sns.barplot(x=ratio_likes_por_categoria.values, y=ratio_likes_por_categoria.index, palette='summer')
plt.xlabel('Ratio de Likes por Visita')
plt.ylabel('Categoría')
plt.title('Ratio de Likes por Categoría')
plot.set_xticklabels(plot.get_xticklabels(), rotation=45)  # Rotar etiquetas
plt.tight_layout()
plt.show()

#D. ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” /“Comentarios”?

# Agrupa por categoría y calcula las sumas de vistas y comentarios
grouped_data = df.groupby('category_id').agg({'views': 'sum', 'comment_count': 'sum'})

grouped_data['views_to_comments_ratio'] = grouped_data['views'] / grouped_data['comment_count']

sorted_data = grouped_data.sort_values(by='views_to_comments_ratio', ascending=False)

best_categories = sorted_data.head(5)

# Imprime los resultados en la consola
print("Mejores Categorías de Videos en función del Ratio Vistas/Comentarios:")
print(best_categories)

# Extrae los datos para el gráfico
categories = best_categories.index
ratios = best_categories['views_to_comments_ratio']

# Crear un gráfico de barras vertical
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, ratios, color='skyblue')
plt.ylabel('Ratio de Vistas / Comentarios')
plt.title('Mejores Categorías de Videos en función del Ratio Vistas/Comentarios')
plt.xlabel('Categoría')

# Agregar los valores del ratio encima de cada barra
for bar, ratio in zip(bars, ratios):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{ratio:.2f}', ha='center', va='bottom')

# Rotar las etiquetas del eje x para mejor legibilidad
plt.xticks(rotation=45, ha='right')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#E. ¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?
# Carga el archivo 'DE_videos.csv' en un DataFrame
df = pd.read_csv('DE_videos.csv')

# Asegura que 'trending_date' sea interpretado como una fecha y hora correctamente
df['trending_date'] = pd.to_datetime(df['trending_date'], errors='coerce')

# Si se logra convertir a fecha, extrae solo la fecha (sin la hora) para la agregación diaria
if pd.api.types.is_datetime64_any_dtype(df['trending_date']):
    df['trending_date'] = df['trending_date'].dt.date

    # Calcula el recuento de video_ids únicos para cada trending_date
    daily_trending_counts = df.groupby('trending_date')['video_id'].nunique().reset_index()
    daily_trending_counts.columns = ['Fecha de tendencia', 'Videos únicos']

    # Gráfico utilizando Seaborn para una visualización más atractiva
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_trending_counts, x='Fecha de tendencia', y='Videos únicos', marker='o', color='purple')
    plt.title('Recuento diario de videos únicos en tendencia con el tiempo')
    plt.xlabel('Fecha de tendencia')
    plt.ylabel('Videos únicos en tendencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Error: No se pudo convertir 'trending_date' a un formato de fecha y hora.")

#F. ¿Qué canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?

# Carga el archivo 'DE_videos.csv' en un DataFrame
df = pd.read_csv('DE_videos.csv')

# Calcula la frecuencia de cada canal en tendencia
channel_trending_frequency = df['channel_title'].value_counts().reset_index()
channel_trending_frequency.columns = ['channel_title', 'trending_frequency']

# Selecciona los 10 principales y los 10 inferiores canales para la visualización
top_channels = channel_trending_frequency.head(10)
bottom_channels = channel_trending_frequency.tail(10)

# Combinando los datos para graficar los 10 principales y 10 inferiores canales
combined_channels = pd.concat([top_channels, bottom_channels])

# Filtra los datos originales solo para los 20 canales seleccionados
filtered_df = df[df['channel_title'].isin(combined_channels['channel_title'])]

# Agrupa los datos filtrados por canal y fecha para contar las tendencias
grouped_data = filtered_df.groupby(['channel_title', 'trending_date']).size().reset_index(name='count')

# Pivotea los datos para crear una tabla que muestre la frecuencia de tendencia por canal y fecha
pivot_data = grouped_data.pivot(index='trending_date', columns='channel_title', values='count').fillna(0)

# Selecciona un rango de fechas para mostrar en el gráfico
date_range = pivot_data.index[::50]  # Mostrar cada 50 días

# Filtra los datos para el rango de fechas seleccionado
pivot_data_filtered = pivot_data.loc[date_range]

# Graficar un gráfico de barras apiladas para mostrar la frecuencia de tendencia por canal y fecha
pivot_data_filtered.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Frecuencia de Tendencia por Canal y Fecha')
plt.xlabel('Fecha de Tendencia')
plt.ylabel('Frecuencia de Tendencia')
plt.legend(title='Canal', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#G. ¿En qué Estados se presenta el mayor número de “Vistas”, “Me gusta” y “No me gusta”?

df = pd.read_csv('DE_videos.csv')

# Agregando los datos por estado y calculando la suma de Vistas, Me gusta y No me gusta.
state_aggregated = df.groupby('state')['views', 'likes', 'dislikes'].sum().reset_index()

# Encontrando los estados con el mayor número de Vistas, Me gusta y No me gusta
highest_views_state = state_aggregated.loc[state_aggregated['views'].idxmax()]
highest_likes_state = state_aggregated.loc[state_aggregated['likes'].idxmax()]
highest_dislikes_state = state_aggregated.loc[state_aggregated['dislikes'].idxmax()]

# Mostrando los resultados
print('State with the highest number of views:\n', highest_views_state)
print('\nState with the highest number of likes:\n', highest_likes_state)
print('\nState with the highest number of dislikes:\n', highest_dislikes_state)

# Configurando el estilo de las gráficas
sns.set_style('whitegrid')

# Creando un gráfico de dispersión con etiquetas para ver las métricas por estado
plt.figure(figsize=(12, 6))
scatterplot = sns.scatterplot(data=state_aggregated, x='views', y='likes', hue='state', size='dislikes', sizes=(20, 200), palette='viridis')

# Añadiendo etiquetas a los puntos para los "dislikes"
for line in range(0, state_aggregated.shape[0]):
    scatterplot.text(state_aggregated['views'][line] + 0.5, state_aggregated['likes'][line], state_aggregated['state'][line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.title('Distribución de Vistas, Me gusta y No me gusta por Estado')
plt.xlabel('Vistas')
plt.ylabel('Me gusta')
plt.legend(title='Estado', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#H. ¿Es factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”?

df = pd.read_csv('DE_videos.csv')

# Seleccionamos las variables
features = ['lat', 'lon']
target = 'dislikes'  # Cambiar el target (views, likes, dislikes)

# Filtra el DataFrame para seleccionar solo las columnas relevantes
data = df[['lat', 'lon', target]]

data = data.dropna()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Crea un modelo de Random Forest Regressor con 100 árboles
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrena el modelo
model.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
predictions = model.predict(X_test)

# Evalúa el rendimiento del modelo utilizando el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, predictions)
print(f'Error Cuadrático Medio (MSE) para "{target}": {mse}')

# Visualiza las predicciones vs. los valores reales
plt.scatter(y_test, predictions)
plt.xlabel(f'{target} Reales')
plt.ylabel(f'{target} Predichos')
plt.title(f'Predicciones vs. Valores Reales ({target})')
plt.show()

#I. ¿Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?

df = pd.read_csv('DE_videos.csv')
# Selecciona las columnas relevantes y elimina filas con valores faltantes
data = df[['views', 'likes']].dropna()

# Calcula la correlación entre 'views' y 'likes'
correlation = data['views'].corr(data['likes'])

# Genera un gráfico de regresión lineal con seaborn
plt.figure(figsize=(8, 6))
sns.regplot(x='views', y='likes', data=data, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
plt.title(f'Relación entre Vistas y Likes (Correlación: {correlation:.2f})')
plt.xlabel('Vistas')
plt.ylabel('Likes')
plt.show()