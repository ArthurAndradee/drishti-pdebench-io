import os
import shutil
from pathlib import Path

def analisar_e_limpar_logs(pasta_ref_str, pasta_alvo_str):
    pasta_ref = Path(pasta_ref_str)
    pasta_alvo = Path(pasta_alvo_str)
    pasta_descartados = pasta_alvo / "descartados"

    # 1. Obter todos os arquivos .darshan
    logs_ref = list(pasta_ref.glob("*.darshan"))
    logs_alvo = list(pasta_alvo.glob("*.darshan"))

    qtd_esperada = len(logs_ref)
    qtd_atual = len(logs_alvo)

    print(f"📊 Análise Inicial:")
    print(f" -> Pasta 1 (Referência): {qtd_esperada} logs")
    print(f" -> Pasta 2 (Alvo): {qtd_atual} logs")

    if qtd_atual <= qtd_esperada:
        print("✅ A pasta alvo já tem a quantidade correta (ou menos). Nenhuma ação necessária.")
        return

    # 2. Estratégia: Filtrar jobs incompletos pelo Tamanho do Arquivo.
    # Em execuções de Machine Learning, jobs completos geram logs pesados (megabytes),
    # enquanto jobs que falham por erro no código ou limite de tempo geram logs leves (alguns KB).
    
    # Criar uma lista de tuplas: (caminho_do_arquivo, tamanho_em_bytes, data_modificacao)
    dados_alvo = []
    for log in logs_alvo:
        status = log.stat()
        dados_alvo.append((log, status.st_size, status.st_mtime))

    # Ordenar do MAIOR para o MENOR tamanho
    dados_alvo.sort(key=lambda x: x[1], reverse=True)

    # Os 40 maiores assumimos que são os jobs que rodaram até o fim (completos)
    logs_completos = dados_alvo[:qtd_esperada]
    logs_falhos = dados_alvo[qtd_esperada:]

    # Validação da heurística para o usuário
    menor_valido_kb = logs_completos[-1][1] / 1024
    maior_invalido_kb = logs_falhos[0][1] / 1024 if logs_falhos else 0

    print("\n🔍 Resultados da Heurística (Tamanho do Darshan Log):")
    print(f" -> Os {qtd_esperada} logs selecionados têm tamanho mínimo de {menor_valido_kb:.2f} KB.")
    print(f" -> Os {len(logs_falhos)} logs rejeitados têm tamanho máximo de {maior_invalido_kb:.2f} KB.")

    if maior_invalido_kb >= menor_valido_kb:
        print("⚠️ CUIDADO: Os tamanhos dos arquivos rejeitados e aceitos se sobrepõem muito. Pode haver falsos positivos.")
    else:
        print("✅ Separação clara de tamanhos! A heurística de jobs falhos funcionou perfeitamente.")

    # 3. Executar a Limpeza
    print(f"\n📁 Movendo {len(logs_falhos)} arquivos extras/falhos para a pasta 'descartados'...")
    pasta_descartados.mkdir(exist_ok=True)

    for log, tamanho, _ in logs_falhos:
        destino = pasta_descartados / log.name
        shutil.move(str(log), str(destino))
        print(f"   [Movido] {log.name} ({tamanho / 1024:.1f} KB)")

    print(f"\n🎉 Concluído! A pasta alvo ({pasta_alvo.name}) agora tem exatamente {qtd_esperada} arquivos, igual à referência.")


if __name__ == "__main__":
    # ==========================================
    # ⚙️ CONFIGURAÇÃO: Insira os caminhos reais
    # ==========================================
    PASTA_1 = "/home/users/aadsilva/ic/erad-2026/darshan-logs/results/unet/burgers/train_models_forward_STD"
    PASTA_2 = "/home/users/aadsilva/ic/erad-2026/darshan-logs/results/unet/burgers/train_models_forward_MPI"
    
    analisar_e_limpar_logs(PASTA_1, PASTA_2)