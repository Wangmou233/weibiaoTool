"""
知识图谱服务
"""

import json
import time
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging
from pathlib import Path

from config.settings import settings
from models.document import Document, DocumentStatus
from models.knowledge import KnowledgeGraph, EntityList,RelationList,Entity,Relation
from utils.logger import get_logger
from utils.api_utils import DeepSeekClient
from api.client import MockAPIClient
from pydantic import BaseModel

logger = get_logger(__name__)

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class KnowledgeService:
    """知识图谱服务"""

    def __init__(self, settings, api_client):
        """
        初始化知识图谱服务

        Args:
            settings: 配置对象
            api_client: API客户端
        """
        self.settings = settings
        self.api_client = api_client
        self.data_dir = settings.data_dir

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化DeepSeek客户端
        if isinstance(api_client, DeepSeekClient):
            self.llm_client = api_client
        elif isinstance(api_client, MockAPIClient):
            # 使用模拟API客户端进行测试
            self.llm_client = api_client
        else:
            # 实际部署时使用DeepSeekClient
            self.llm_client = DeepSeekClient(settings)

    def extract_knowledge_from_document(
        self,
        document: Document,
        timeout: Optional[int] = None
    ) -> KnowledgeGraph:
        """
        从文档中抽取知识

        Args:
            document: 文档对象
            timeout: 超时时间（秒）

        Returns:
            抽取的知识图谱
        """
        if timeout is None:
            timeout = self.settings.default_timeout

        # 更新文档状态为处理中
        document.update_status(DocumentStatus.PROCESSING)
        self._save_document(document)

        try:
            # 读取文档内容
            content = self._read_document_content(document)

            # 使用大模型抽取知识
            kg = self._extract_knowledge_with_llm(content, document.id)

            # 更新文档状态为已完成
            document.update_status(DocumentStatus.COMPLETED)
            document.update_processing_result(
                entities=[e.to_dict() for e in kg.entities],
                relations=[r.to_dict() for r in kg.relations],
                processing_time=time.time() - document.updated_at.timestamp()
            )

            # 保存文档更新
            self._save_document(document)

            logger.info(f"知识抽取完成: {document.name}")
            return kg

        except Exception as e:
            # 更新文档状态为错误
            document.update_status(DocumentStatus.ERROR, str(e))
            self._save_document(document)

            logger.error(f"知识抽取失败: {document.name} - {str(e)}")
            raise

    def extract_knowledge_from_documents(
        self,
        document_ids: List[str],
        concurrency: int = None,
        timeout: Optional[int] = None
    ) -> Dict[str, KnowledgeGraph]:
        """
        从多个文档中批量抽取知识

        Args:
            document_ids: 文档ID列表
            concurrency: 并发数量
            timeout: 超时时间（秒）

        Returns:
            文档ID到知识图谱的映射
        """
        if concurrency is None:
            concurrency = self.settings.max_concurrency

        if timeout is None:
            timeout = self.settings.default_timeout

        results = {}
        failed = []

        # 获取文档对象
        documents = []
        for doc_id in document_ids:
            doc = self._get_document(doc_id)
            if doc:
                documents.append(doc)
            else:
                logger.error(f"文档不存在: {doc_id}")
                failed.append(doc_id)

        # 批量处理文档
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交任务
            futures = {
                executor.submit(self.extract_knowledge_from_document, doc, timeout): doc.id
                for doc in documents
            }

            # 收集结果
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    kg = future.result()
                    results[doc_id] = kg
                except Exception as e:
                    logger.error(f"文档处理失败: {doc_id} - {str(e)}")
                    failed.append(doc_id)

        logger.info(f"批量处理完成: 成功 {len(results)}, 失败 {len(failed)}")
        return results

    def save_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """
        保存知识图谱

        Args:
            kg: 知识图谱
        """
        kg_path = self.data_dir / f"kg_{kg.id}.json"

        try:
            with open(kg_path, 'w', encoding='utf-8') as f:
                f.write(kg.json())

            logger.info(f"知识图谱保存成功: {kg.name}")
        except Exception as e:
            logger.error(f"保存知识图谱失败: {str(e)}")
            raise

    def load_knowledge_graph(self, kg_id: str) -> Optional[KnowledgeGraph]:
        """
        加载知识图谱

        Args:
            kg_id: 知识图谱ID

        Returns:
            知识图谱对象，不存在则返回None
        """
        kg_path = self.data_dir / f"kg_{kg_id}.json"

        if not kg_path.exists():
            return None

        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                data = f.read()
                return KnowledgeGraph.from_dict(data)
        except Exception as e:
            logger.error(f"加载知识图谱失败: {str(e)}")
            return None

    def merge_knowledge_graphs(
        self,
        kg_ids: List[str],
        merge_entities: bool = True,
        merge_relations: bool = True
    ) -> Optional[KnowledgeGraph]:
        """
        合并多个知识图谱

        Args:
            kg_ids: 知识图谱ID列表
            merge_entities: 是否合并实体
            merge_relations: 是否合并关系

        Returns:
            合并后的知识图谱，失败则返回None
        """
        # 加载所有知识图谱
        kgs = []
        for kg_id in kg_ids:
            kg = self.load_knowledge_graph(kg_id)
            if kg:
                kgs.append(kg)

        if not kgs:
            return None

        # 合并知识图谱
        merged = kgs[0]
        for kg in kgs[1:]:
            merged = merged.merge_with(kg, merge_entities, merge_relations)

        # 保存合并后的知识图谱
        merged.id = f"merged_{int(time.time())}"
        merged.name = f"合并图谱_{len(kgs)}个"
        merged.updated_at = datetime.now()

        self.save_knowledge_graph(merged)
        logger.info(f"知识图谱合并完成: {merged.name}")

        return merged

    def _read_document_content(self, document: Document) -> str:
        """
        读取文档内容

        Args:
            document: 文档对象

        Returns:
            文档内容
        """
        file_path = Path(document.file_path)
        if not file_path.exists():
            raise ValueError(f"文档文件不存在: {file_path}")

        try:
            # 统一通过工具读取文本（兼容 .doc/.docx）
            from utils.word_reader import read_word_text

            text = read_word_text(file_path)
            return text
        except Exception as e:
            logger.error(f"读取文档内容失败: {str(e)}")
            raise
    
    def _extract_knowledge_with_llm(self, content: str, source_doc_id: str) -> KnowledgeGraph:
        """
        使用大模型从内容中抽取知识

        Args:
            content: 文档内容
            source_doc_id: 来源文档ID

        Returns:
            抽取的知识图谱
        """

        parser_Entity = PydanticOutputParser(pydantic_object=EntityList)

        # 获取 LLM 提示中要求的输出格式说明
        format_instructions = parser_Entity.get_format_instructions()

        # 构建提示词
        
        prompt_Entity = self.build_entity_extraction_prompt(content,format_instructions,source_doc_id)
        # 调用大模型
        try:
            response_Entity = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱工程师，擅长从非结构化文本中抽取结构化信息，并转换为 Neo4j 图数据库的 Cypher 查询语言。"},
                    {"role": "user", "content": prompt_Entity}
                ],
                temperature=0.2,
                max_tokens=8000
            )
            
        except Exception as e:
            logger.error(f"实体抽取失败: {str(e)}")
            # 这里不能直接更新文档状态，因为document变量不在这个方法的参数中
            # 返回一个空的知识图谱
            return KnowledgeGraph(
                id=f"error_{source_doc_id}",
                name=f"Error Knowledge Graph for {source_doc_id}",
                entities=[],
                relations=[],
                metadata={"error": f"大模型调用失败: {str(e)}"}
            )

        # 解析响应
        try:
            result_text_entity = response_Entity["choices"][0]["message"]["content"]
            
            # 确保文本是UTF-8编码
            if isinstance(result_text_entity, str):
                result_text = result_text_entity.encode('utf-8', errors='ignore').decode('utf-8')
              
        except (KeyError, IndexError, UnicodeError) as e:
            logger.error(f"解析大模型响应失败: {str(e)}")
            # 返回一个空的知识图谱
            return KnowledgeGraph(
                id=f"error_{source_doc_id}",
                name=f"Error Knowledge Graph for {source_doc_id}",
                entities=[],
                relations=[],
                metadata={"error": f"解析响应失败: {str(e)}"}
            )
        parser_Relation = PydanticOutputParser(pydantic_object=RelationList)

        # 获取 LLM 提示中要求的输出格式说明
        format_instructions_relation = parser_Relation.get_format_instructions()

        # 构建提示词
        result_text = result_text_entity.strip()
        if result_text.startswith("```"):
            result_text = result_text.strip("`")
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()

        # 转换为 Python 对象
        data_entity = json.loads(result_text)

        # 只保留部分字段
        filtered_data = [
            {
                "id":  item.get('id') ,
                "name": item.get("name"),
                "type": item.get("type"),
            }
            for item in data_entity
        ]
        
        prompt_Relation = self.build_relation_extraction_prompt(content,filtered_data,format_instructions_relation)

        try:
            response_Relation = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱工程师，擅长从非结构化文本中抽取结构化信息，并转换为 Neo4j 图数据库的 Cypher 查询语言。"},
                    {"role": "user", "content": prompt_Relation}
                ],
                temperature=0.2,
                max_tokens=8000
            )
            
        except Exception as e:
            logger.error(f"关系抽取失败: {str(e)}")
            # 这里不能直接更新文档状态，因为document变量不在这个方法的参数中
            # 返回一个空的知识图谱
            return KnowledgeGraph(
                id=f"error_{source_doc_id}",
                name=f"Error Knowledge Graph for {source_doc_id}",
                entities=[],
                relations=[],
                metadata={"error": f"大模型调用失败: {str(e)}"}
            )

        # 解析响应
        try:
            result_text_relation = response_Relation["choices"][0]["message"]["content"]
            # 确保文本是UTF-8编码
            if isinstance(result_text, str):
                result_text = result_text.encode('utf-8', errors='ignore').decode('utf-8')
               
        except (KeyError, IndexError, UnicodeError) as e:
            logger.error(f"解析大模型响应失败: {str(e)}")
            # 返回一个空的知识图谱
            return KnowledgeGraph(
                id=f"error_{source_doc_id}",
                name=f"Error Knowledge Graph for {source_doc_id}",
                entities=[],
                relations=[],
                metadata={"error": f"解析响应失败: {str(e)}"}
            )
        result_text = result_text_relation.strip()
        if result_text.startswith("```"):
            result_text = result_text.strip("`")
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()

        # 转换为 Python 对象
        data_relation = json.loads(result_text)
        print(data_relation)
        cypher_statements =self._generate_cypher(data_entity,data_relation,source_doc_id)
        kg_data = KnowledgeGraph(
                id=f"kg_{source_doc_id}_{int(time.time())}",
                name=f"知识图谱_{source_doc_id}",
                entities=[Entity.from_dict(e) for e in data_entity],
                relations=[Relation.from_dict(a) for a in data_relation],
                description=f"从文档 {source_doc_id} 中抽取的知识",
                metadata={
                "source_document": source_doc_id,
                "cypher_statements": cypher_statements
            }
            )

        return kg_data


    def _generate_cypher(self, entities: List[Dict], relations: List[Dict],doc_id) -> List[str]:
        """
        根据实体和关系生成 Neo4j 的 Cypher 语句

        Args:
            entities: List[Dict]，每个字典至少包含 id, name, type, properties
            relations: List[Dict]，每个字典至少包含 source_id, target_id, type
            doc_id: 来源文档ID

        Returns:
            List[str]，Cypher 语句列表
        """
        cypher_statements = []

        for e in entities:
            node_label = e['type'].capitalize()
            props = e.get("properties", {}).copy()
            props["name"] = e["name"]  # 把name也放到属性里
            # 转换为 Cypher Map 格式
            props_text = ", ".join(f"{k}: '{v}'" for k, v in props.items())
            
            cypher = f"MERGE (n:{node_label} {{id: '{e['id']}_{doc_id}'}}) ON CREATE SET n += {{{props_text}}};"
            cypher_statements.append(cypher)



        # 生成关系
        for r in relations:
            rel_type = r['type'].upper()  # 关系类型大写
            cypher = (
                f"MATCH (a {{id: '{r['source']}_{doc_id}'}}), (b {{id: '{r['target']}_{doc_id}'}}) "
                f"MERGE (a)-[:{rel_type}]->(b);"
                
            )
            cypher_statements.append(cypher)

        return cypher_statements

    def build_entity_extraction_prompt(self,content: str, format_instructions: str,source_doc_id: str) -> str:
        return f"""
    你是一个知识图谱实体抽取助手。请从以下文本中严格抽取**实体**，不需要关系。
    ## 文档结构指南：
    1. 项目基本信息：名称、编码、工期、危险源、动火等级等通常在文档顶部
    2. 资源需求：工具列表通常标记为"需用工机具"，材料列表标记为"需用材料"
    3. 人员配置：通常在"需要人数"或"工种代码"部分
    4. 活动流程：按事件顺序排列的活动，包含描述、工时、作业方法和安全措施,并包括每个工序的所用材料和人员与工具，这是非常重要的关系.
    5. 危险源也作为单独节点。

    ## 要求：
    - 仅输出实体，不包含关系。
    - 每个实体必须有唯一 id,其格式为(节点的type)_(数字)。
    - 必须标注实体类型（如 工具、材料、人员、活动、项目、危险源（hazrad）等）。
    - 实体属性必须尽量完整，例如：名称、数量、单位、代码等。
    - 工人应当属于role类型中
    - 注意对后续步骤节点的提取
    - 工序节点的properties要包含工时(duration),step,technology_point,safety_measures,还有作业方法。
    - 注意每个工人应当有单独的节点来进行描述。
    - 节点的属性不应包括符号：'[',']','{','}','''等。
    - 输出必须符合以下格式：

    {format_instructions}

    ## 文档内容：
    {content}
    """

    def build_relation_extraction_prompt(self,content: str,entities_list: str ,format_instructions: str) -> str:
        return f"""
    你是一个专业的知识图谱工程师，擅长从非结构化文本中抽取结构化关系信息。
    请根据以下条件从给定文本中抽取实体之间的关系：

    1. 已知实体列表（包含实体ID、名称和类型）：
    {entities_list}

    2. 关系类型及对应关键词：
    part_of, used_in, related_to, causes, is_solution_for, requires, results_in,
    checked_by, maintained_by, follows, depends_on, compiled_by, created_by,
    operated_by, located_in, contains, belongs_to, connects_to, replaces,
    improves, prevents, attention_to ,other

    ## 文档结构指南：
    1. 项目基本信息：名称、编码、工期、危险源、动火等级等通常在文档顶部
    2. 资源需求：工具列表通常标记为"需用工机具"，材料列表标记为"需用材料"
    3. 人员配置：通常在"需要人数"或"工种代码"部分
    4. 活动流程：按事件顺序排列的活动，包含描述、工时、作业方法和安全措施,并包括每个工序的所用材料和人员与工具，这是非常重要的关系.

    3. 输出要求：
    - 仅输出 JSON 数组，格式为：
    {format_instructions}
    - 不要输出额外说明文字。
    - 如果文本中没有关系，则输出空数组 []。
    - type 必须是上面列出的关系类型之一
    - 其中每个工序应当与其对应的人员（operated_by）、材料（used_in）、工具（used_in），建立所属关系
    - 工序和工序之间有先后关系（follows） 
    - 项目与工具的关联（used_in）、材料的关联（used_in）、人员的关联（operated_by）
    - 项目与危险源的关联（attention_to）
    4. 文本内容：
    {content}

    请根据文本和实体列表抽取所有明确的关系。

    """

    def _build_extraction_prompt(self, content: str, format_instructions: str) -> str:
        """
        构建知识抽取提示词

        Args:
            content: 文档内容
            format_instructions: 输出格式

        Returns:
            提示词
        """
        return f"""
    你正在解析工业设备维修作业标准文档的。请从以下文本中抽取结构化知识，并构建一个完整的知识图谱。

    ## 文档结构指南：
    1. 项目基本信息：名称、编码、工期、危险源、动火等级等通常在文档顶部。
    2. 资源需求：工具列表通常标记为"需用工机具"，材料列表标记为"需用材料"。
    3. 人员配置：通常在"需要人数"或"工种代码"部分。
    4. 活动流程：按事件顺序排列的活动，包含描述、工时、作业方法和安全措施并包含一个工序所需的人员和材料，这是非常重要的关系。

    ## 特殊处理规则：
    - 工具ID格式：字母编号（如 A、B、C）或数字编号。
    - 工时单位统一转换为小时（h）。
    - 日期格式统一为 YYYY-MM-DD。
    - 人员代码保留原始格式（如 QG1、QZG2）。

    ## 数据验证要求：
    - 工具/材料引用必须存在于资源列表中。
    - 所有实体必须有唯一 ID。
    - 每个关系的 source 和 target 必须指向 entities 中的合法 ID。

    ## 关系完整性要求（重点）：
    - 所有实体之间的依赖关系必须完整解析，不允许遗漏。
    - 至少包括以下常见关系：
    - 项目需要工具/材料/人员（require）
    - 工序依赖前序工序（depends_on）
    - 工序与项目的关联（part_of）
    - 工序与材料工具人员的关联（used_in）
    - 项目与危险源的关联（attention_to）
    - 如果发现潜在的依赖或引用关系，必须显式写入 relations。

    ## 关系构建硬性要求：
    1. 任何实体如果在文本中存在依赖、引用或使用关系，必须显式生成一个 Relation。
    2. Relation 的 source 和 target 必须对应 entities 中已存在的 ID。
    3. 每个工序（procedure）必须至少有：
    - 一个 "part_of" 指向所属项目
    - 所需的工具/材料/人员的 "requires" 或 "used_in"
    - 前序工序（如果有）的 "depends_on"
    4. 如果文本中隐含了引用关系（例如工序提到“使用工具 A”），也必须转化为一个 Relation。
    5. 如果不确定关系类型，使用 "related_to"，绝对不能省略。

    ## 输出格式要求：
    严格输出为以下格式（可直接被解析为 KnowledgeGraph 对象）：
    {format_instructions}

    文档内容：
    {content}
    """

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析大模型响应

        Args:
            response_text: 响应文本

        Returns:
            解析后的数据
        """
        logger.info(f"开始解析LLM响应，响应长度: {len(response_text)}")
        logger.debug(f"响应内容前500字符: {response_text[:500]}")
        
        try:
            # 尝试直接解析JSON
            data = json.loads(response_text)
            logger.info(f"成功解析JSON，包含实体: {len(data.get('entities', []))}，关系: {len(data.get('relations', []))}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"直接JSON解析失败: {str(e)}，尝试提取JSON部分")
            
            # 如果不是有效的JSON，尝试提取JSON部分
            import re

            # 使用更复杂的正则表达式来匹配完整的JSON对象
            json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if not matches:
                # 尝试寻找代码块中的JSON
                code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
                if code_matches:
                    matches = code_matches

            if not matches:
                logger.error("无法从响应中提取JSON")
                logger.debug(f"完整响应内容: {response_text}")
                return {"entities": [], "relations": [], "cypher_statements": []}

            # 尝试解析找到的JSON字符串
            for json_str in matches:
                try:
                    data = json.loads(json_str)
                    logger.info(f"成功从文本中提取并解析JSON，包含实体: {len(data.get('entities', []))}，关系: {len(data.get('relations', []))}")
                    return data
                except json.JSONDecodeError:
                    continue
            
            logger.error("所有JSON提取尝试都失败")
            logger.debug(f"找到的潜在JSON字符串: {matches}")
            return {"entities": [], "relations": [], "cypher_statements": []}

    def _save_document(self, document: Document) -> None:
        """
        保存文档信息

        Args:
            document: 文档对象
        """
        from services.document_service import DocumentService

        # 使用文档服务保存文档
        doc_service = DocumentService(self.settings, self.api_client)
        doc_service._save_document(document)

    def _get_document(self, document_id: str) -> Optional[Document]:
        """
        获取文档对象

        Args:
            document_id: 文档ID

        Returns:
            文档对象，不存在则返回None
        """
        from services.document_service import DocumentService

        # 使用文档服务获取文档
        doc_service = DocumentService(self.settings, self.api_client)
        return doc_service.get_document(document_id)
