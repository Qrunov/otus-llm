#!/usr/bin/env python3
"""
Скрипт для оценки качества ответов vLLM с помощью GigaChat
"""

import os
import glob
import requests
import mlflow
import json
from pathlib import Path
from typing import List, Tuple
import time
from datetime import datetime
import uuid

class VLLMClient:
    """Клиент для взаимодействия с vLLM"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def explain_code(self, code: str, model: str = "meta-llama/Llama-3.1-8B-Instruct") -> str:
        """Отправляет код на объяснение в vLLM"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful C++ assistant"},
                {"role": "user", "content": f"Explain me this code:\n\n{code}"}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Ошибка при обращении к vLLM: {e}")
            return ""


class GigaChatClient:
    """Клиент для взаимодействия с GigaChat API"""
    
    def __init__(self, credentials: str, scope: str = "GIGACHAT_API_PERS"):
        self.credentials = credentials
        self.scope = scope
        self.access_token = None
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self._get_token()
    
    def _get_token(self):
        """Получает токен доступа"""
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        headers = {
            "Authorization": f"Basic {self.credentials}",
            'Accept': 'application/json',
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"scope": self.scope}
        
        try:
            response = requests.post(url, headers=headers, data=data, verify=False, timeout=30)
            response.raise_for_status()
            self.access_token = response.json()['access_token']
        except Exception as e:
            print(f"Ошибка получения токена GigaChat: {e}")
            raise
    
    def evaluate_response(self, original_code: str, llm_explanation: str) -> int:
        """Оценивает корректность ответа локальной LLM"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Оригинальный код C++:
{original_code}

Объяснение от локальной LLM:
{llm_explanation}

Оцени корректность объяснения локальной LLM от 1 до 10. Отвечай кратко: в ответ только число."""
        
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, verify=False, timeout=30)
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            # Извлекаем число из ответа
            score = int(''.join(filter(str.isdigit, answer)))
            return min(max(score, 0), 10)  # Ограничиваем 0-10
        except Exception as e:
            print(f"Ошибка при оценке через GigaChat: {e}")
            return 0


class LLMEvaluator:
    """Основной класс для оценки качества работы LLM"""
    
    def __init__(self, cpp_dir: str, vllm_url: str, gigachat_creds: str, 
                 vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.cpp_dir = cpp_dir
        self.vllm_client = VLLMClient(vllm_url)
        self.gigachat_client = GigaChatClient(gigachat_creds)
        self.vllm_model = vllm_model
        self.results = []
    
    def get_cpp_files(self) -> List[str]:
        """Получает список C++ файлов из каталога"""
        patterns = ['*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp']
        files = []
        
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.cpp_dir, pattern)))
            files.extend(glob.glob(os.path.join(self.cpp_dir, '**', pattern), recursive=True))
        
        return sorted(set(files))
    
    def read_file(self, filepath: str) -> str:
        """Читает содержимое файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def evaluate_file(self, filepath: str) -> Tuple[str, str, int]:
        """Оценивает один файл"""
        print(f"\n📄 Обработка: {filepath}")
        
        # Читаем код
        code = self.read_file(filepath)
        
        # Получаем объяснение от vLLM
        print("  ⏳ Запрос к vLLM...")
        explanation = self.vllm_client.explain_code(code, self.vllm_model)
        
        if not explanation:
            print("  ❌ Не удалось получить ответ от vLLM")
            return filepath, "", 0
        
        print(f"  ✓ Получен ответ от vLLM ({len(explanation)} символов)")
        
        # Получаем оценку от GigaChat
        print("  ⏳ Запрос к GigaChat для оценки...")
        score = self.gigachat_client.evaluate_response(code, explanation)
        
        print(f"  ✓ Оценка: {score}/10")
        
        return filepath, explanation, score
    
    def run_evaluation(self, experiment_name: str = "vLLM_Evaluation"):
        """Запускает полную оценку"""
        files = self.get_cpp_files()
        
        if not files:
            print(f"⚠️  Не найдено C++ файлов в {self.cpp_dir}")
            return
        
        print(f"🚀 Найдено {len(files)} C++ файлов")
        print(f"📊 MLflow эксперимент: {experiment_name}")
        
        # Настройка MLflow
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логируем параметры
            mlflow.log_param("vllm_url", self.vllm_client.base_url)
            mlflow.log_param("vllm_model", self.vllm_model)
            mlflow.log_param("cpp_directory", self.cpp_dir)
            mlflow.log_param("files_count", len(files))
            
            scores = []
            
            # Обрабатываем каждый файл
            for i, filepath in enumerate(files, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(files)}] {Path(filepath).name}")
                print('='*60)
                
                try:
                    filename, explanation, score = self.evaluate_file(filepath)
                    scores.append(score)
                    
                    # Логируем метрики для каждого файла
                    mlflow.log_metric(f"score_{Path(filepath).stem}", score, step=i)
                    
                    self.results.append({
                        'file': filename,
                        'explanation': explanation,
                        'score': score
                    })
                    
                except Exception as e:
                    print(f"  ❌ Ошибка при обработке файла: {e}")
                    scores.append(0)
                
                # Небольшая задержка между запросами
                time.sleep(1)
            
            # Вычисляем итоговые метрики
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                print(f"\n{'='*60}")
                print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
                print('='*60)
                print(f"Средняя оценка: {avg_score:.2f}/10")
                print(f"Максимальная оценка: {max_score}/10")
                print(f"Минимальная оценка: {min_score}/10")
                print(f"Всего файлов: {len(scores)}")
                
                # Логируем итоговые метрики
                mlflow.log_metric("average_score", avg_score)
                mlflow.log_metric("max_score", max_score)
                mlflow.log_metric("min_score", min_score)
                mlflow.log_metric("total_files", len(scores))
                
                # Сохраняем детальные результаты
                results_file = "/tmp/evaluation_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                
                mlflow.log_artifact(results_file, "results")
                
                return avg_score
            else:
                print("⚠️  Нет результатов для обработки")
                return 0.0


def main():
    """Главная функция"""
    
    # ============= КОНФИГУРАЦИЯ =============
    # Укажите ваши параметры здесь
    
    CPP_DIRECTORY = "samples"  # Путь к каталогу с C++ файлами
    VLLM_URL = "http://localhost:8000/v1"  # URL вашего vLLM сервера
    VLLM_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Модель в vLLM
    GIGACHAT_CREDENTIALS = "type here your credential"  # Credentials для GigaChat
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # URI для MLflow
    
    # Можно использовать переменные окружения
    CPP_DIRECTORY = os.getenv("CPP_DIR", CPP_DIRECTORY)
    VLLM_URL = os.getenv("VLLM_URL", VLLM_URL)
    VLLM_MODEL = os.getenv("VLLM_MODEL", VLLM_MODEL)
    GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDS", GIGACHAT_CREDENTIALS)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    
    # =========================================
    
    # Настройка MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Отключаем предупреждения SSL для GigaChat
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Создаем и запускаем оценщик
    evaluator = LLMEvaluator(
        cpp_dir=CPP_DIRECTORY,
        vllm_url=VLLM_URL,
        gigachat_creds=GIGACHAT_CREDENTIALS,
        vllm_model=VLLM_MODEL
    )
    
    evaluator.run_evaluation()
    
    print(f"\n✅ Оценка завершена!")
    print(f"📊 Результаты сохранены в MLflow: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
