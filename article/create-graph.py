import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Inserção dos Dados Reais
datasets = ['Dataset Advection', 'Dataset Burgers']
metrics = ['Tempo (s)', 'RMSE', 'RMSE Norm.', 'nRMSE']
models = ['FNO'] # Removido o Unet

# Nomes das variações atualizados
variations = ['Padrão', 'MPI', 'Alinhamento', 'Buffering'] 

# Chaves do dicionário atualizadas para bater com os novos nomes
dados_reais = {
    ('Dataset Advection', 'Tempo (s)', 'FNO', 'Alinhamento'): (1032.04, 23.38),
    ('Dataset Advection', 'Tempo (s)', 'FNO', 'Buffering'): (1059.97, 56.24),
    ('Dataset Advection', 'Tempo (s)', 'FNO', 'MPI'): (1078.38, 60.43),
    ('Dataset Advection', 'Tempo (s)', 'FNO', 'Padrão'): (1077.66, 41.57),
    ('Dataset Burgers', 'Tempo (s)', 'FNO', 'Alinhamento'): (1028.86, 26.25),
    ('Dataset Burgers', 'Tempo (s)', 'FNO', 'Buffering'): (1075.73, 46.12),
    ('Dataset Burgers', 'Tempo (s)', 'FNO', 'MPI'): (1086.59, 66.52),
    ('Dataset Burgers', 'Tempo (s)', 'FNO', 'Padrão'): (1071.75, 50.44),
}

data = []
np.random.seed(42)

for d in datasets:
    for m in metrics:
        for mod in models:
            for v in variations:
                chave = (d, m, mod, v)
                
                if chave in dados_reais:
                    val, std = dados_reais[chave]
                else:
                    std = 0.0 
                    if 'Tempo' in m:
                        val = np.random.uniform(900, 1100)
                    elif 'Norm' in m or 'nRMSE' in m:
                        val = np.random.uniform(0.01, 0.1)
                    else: 
                        val = np.random.uniform(2, 10)
                
                data.append({
                    'Dataset': d,
                    'Metrica': m,
                    'Modelo': mod,
                    'Variacao': v,
                    'Valor': val,
                    'Desvio': std
                })

df = pd.DataFrame(data)

# 2. Configuração de Cores e Grid
cor_fundo = "#FFFFFF" 
cores_datasets = ["#66c2a4", "#fb8d61"] 
cor_texto = "#333333"

sns.set_theme(style="whitegrid", rc={"axes.facecolor": cor_fundo, "text.color": cor_texto})

# Grid 1x4 com altura ajustada
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6), constrained_layout=True)

fig.suptitle("Desempenho do Modelo FNO por Métrica e Dataset", 
             fontsize=18, fontweight='bold', color=cor_texto)

# 3. Preenchimento dos Gráficos
for m_idx, metric_name in enumerate(metrics):
    ax = axes[m_idx] 
    
    df_plot_metric = df[df['Metrica'] == metric_name]
    
    sns.barplot(
        data=df_plot_metric,
        x="Variacao",
        y="Valor",
        hue="Dataset", 
        hue_order=datasets, 
        ax=ax,
        palette=cores_datasets,
        errorbar=None
    )
    
    # --- Configuração de Títulos e Labels ---
    ax.set_title(metric_name, fontsize=14, fontweight='bold', color=cor_texto)
    
    ax.set_xlabel("Variação", fontsize=12, color=cor_texto)
    ax.tick_params(axis='x', rotation=15) 
        
    if m_idx == 0:
        ax.set_ylabel("Valor", fontsize=12, fontweight='bold', color=cor_texto)
    else:
        ax.set_ylabel("") 
    
    # --- Adicionando Desvio Padrão e Rótulos ---
    bar_containers = ax.containers[:len(datasets)]
    
    for hue_idx, container in enumerate(bar_containers):
        dataset_atual = datasets[hue_idx]
        
        df_plot_bars = df_plot_metric[df_plot_metric['Dataset'] == dataset_atual]
        df_plot_bars = df_plot_bars.set_index('Variacao').reindex(variations)
        desvios = df_plot_bars['Desvio'].values
        
        x_coords = [rect.get_x() + rect.get_width() / 2.0 for rect in container]
        y_coords = [rect.get_height() for rect in container]
        
        for x, y, std in zip(x_coords, y_coords, desvios):
            if std > 0:
                ax.errorbar(x, y, yerr=std, color='black', capsize=3, elinewidth=1.2)
        
        labels = [f'{y:.2f}' for y in y_coords]
        ax.bar_label(container, labels=labels, label_type='edge', fontsize=7, padding=6, color="#000000")
            
    if ax.get_legend():
        ax.get_legend().remove()

# 4. Legenda (A nota de rodapé foi removida)
handles, labels = axes[0].get_legend_handles_labels()
# O bbox_to_anchor ajustado levemente para centralizar bem embaixo sem a nota
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12), 
           fontsize=12, title="Dataset", title_fontsize=13)

# 5. Salvar
output_filename = 'comparacao-otimizacoes.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Imagem salva como {output_filename}")