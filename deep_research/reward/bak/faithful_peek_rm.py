import re
from typing import List
import os
import json
import uuid
import json
from urllib.parse import quote
import math
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# https://internal.invalid/ray-project/serving-manual/bxwg1egx6poxdgoz
# pip install rserving_client -i https://internal.invalid/simple
from rserving_client.rserving_client import (
    ArksConfig,
    ArksRequest,
    MayaClient,
    RouterPriority,
    LoadBalanceStrategy,
)


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class Map_faithful_model_based_openai(ORM):
    def __init__(
        self,
    ):
        self.ray_client = MayaClient.get_instance()

    def test_ray(self, infer_data):
        self.ray_request = ArksRequest()
        self.ray_request.scene_name = "finance_yanjin_judge_a100"
        self.ray_request.chain_name = "v1"
        random_str = str(uuid.uuid4())
        # 去掉连字符
        session_id = str(uuid.uuid4()).replace("-", "")
        # print(session_id)
        self.ray_request.request_timeout = 60000
        self.ray_request.read_timeout = 60000
        self.ray_request.connect_timeout = 60000
        self.ray_request.load_balancing_policy = LoadBalanceStrategy.RANDOM
        # self.ray_request.session_id = "12424"
        self.ray_request.session_id = session_id
        self.ray_request.set_lbconfig({"arks.antvip.arrow.cross.city": "true"})
        self.ray_request.set_item_num(1)
        self.ray_request.router_priority = RouterPriority.CONFIG
        self.ray_request.set_serverlist(["33.157.9.50:10001"])
        # self.ray_request.item_number_per_packet = 1 # 发给每个后端请求的item个数
        # self.ray_request.max_packet_count = 1 # 这个request最多分为多少个发给后端的请求包
        # item_id = "123456789"
        # 生成 UUID
        random_str = str(uuid.uuid4())
        # 去掉连字符
        item_id = str(uuid.uuid4()).replace("-", "")
        # print(item_id)

        item = self.ray_request.add_item(item_id)
        tensor_key = "data"
        tensor_values = [json.dumps(infer_data, ensure_ascii=False)]
        # print("输入数据", tensor_values)
        shapes = [1]
        item.add_tensor_feature(tensor_key, shapes, tensor_values)
        try:
            res = self.ray_client.call(self.ray_request)
            if res.success:
                print("request success")
                result_dict = res.items[0].attributes
                result_json_data = json.loads(result_dict["result"])
                content = result_json_data["choices"][0]["message"]["content"]
                print(content)
                return content
            else:
                print("request fail")
                print(res.error_msg)
                return ""
        except:
            return ""

    def build_extract_template(self, prompt, response):
        prompt_template = """# 任务(提取元组后判断类型和原因)
    你现在在进行金融信息提取和判断能力的考试，你需要经过严密的分析思考完成考试，考试分为两个步骤(最终结果按# 输出格式要求 输出)：
    第一步，提取出<回复答案>中所有和金融数值相关的四元组信息（主体，指标，数值，时间）；第二步，判断提取的四元组信息是否和# 参考材料中表述的信息一致。

    # 金融四元组的定义
    1. 主体：指标的主体，比如：XXX基金/股票/债券、XXX公司/行业、板块、人名、产品名等，保持提取的主体完整性和明确性
    2. 指标：金融指标的名称，详见**金融指标字典**
    3. 数值：指标的数值描述，可带正负号、单位、中文、千位逗号、比较词低于/高于/不少于等，比如：3.14、-0.5%、1万元、2,000手，高于1.3%等，保持提取的数值准确性
    4. 时间：指标的时效性，金融数值的时间属性，通常关联到一个特定的时间点、或者由起始和结束的时间段组成，比如：2025年03月04日，时间段2024年初至今

    # step1:提取金融四元组的具体要求
    1. 你需要对**每个**<回复答案>中的内容依次进行分析，对其中所有包含**数值**的信息提取出四元组信息。
    2. 提取的指标应该在**金融指标字典**内，或者是与金融指标字典相关的指标名称。
    3. 在满足回答要求2的基础上，注意每遗漏一个**数值**都会被扣分，请仔细作答。

    ## 金融指标字典
    1. 通用指标：[价格、涨跌额、涨跌幅、开盘价、收盘价、最高价、最低价、均价、当前价、XXX价、收益率、收益金额、年化收益率、累积收益率、持有收益率、超额收益率、XXX收益率、超额回撤、动态回撤、最大回撤、ROE、ROA、资金流入、资金流出、资金净流入、资金净流出、久期、胜率、赔率、XXX年利率、XXX额度、成交量、成交额、融资净买入、公允价值]
    2. 持仓指标：[配置比例、建议配置比例、持仓收益影响占比、持有金额、持有收益金额]
    3. 基金指标：[份额、净值、持仓前十占比、综合费率、托管费率、管理费率、基金代码、万份收益、单位净值、7日年化收益率、规模、跟踪误差、夏普比率、XXX年化收益率]
    4. 基金经理指标：[任期回报、年化回报、在管天数、管理规模、净值回报]
    5. 股票指标：[基本每股收益、每股净资产、主力资金净流入、游资资金净流入、散户资金净流入、超大单净流入、大单净流入、中小单净流入、换手率、流通股数、总股数、市值、市盈率、市净率、支撑位价格、目标价、每股净资产]
    6. 财报指标：[营业收入、净利润、经营活动产生的现金流量净额、研发费用、营业收入、资产负债率、毛利润、毛利率、流动负债、归母净利润、主营收入、扣非净利润、流动比率、速动比率、EBITDA、调整后EBITDA、业务收入、总收入、每股经营现金流、营销费用]
    7. 前缀：[平均、预期、建议、滚动]
    7.1 * 可拼接在1-6中列举的指标的前面，形成新的指标
    8. 后缀：[同比上升/上涨/增长、环比上升/上涨/增长、xx天移动平均、历史分位数、增速、中位数、排名]
    8.1 * 可拼接在1-6中列举的指标的后面，形成新的指标

    ## 提取金融四元组的特殊说明,你在考试时需要注意以下特殊情形
    -- 对与举例类型的数值，通常是假定/假设类信息，因此与之关联的数值提取的**时间**信息，需要提取为“举例假设信息无需明确时效”，涉及的关键词【当XXX、假设XXX、如果XXX、假定XXX、如XXX、比如XXX、例如XXX、若XXX、等】。
    -- 提取的**指标**在# 金融指标字典找不到或者存在模糊对应关系时，尽可能保留<回复答案>中原始指标名称。
    -- 你需要提取XXX股票/基金代码，股票/基金代码的**数值**通常由6位连续数字组成，通常有对应的股票/基金名，提取**指标**股票代码/基金代码时，提取的**时间**为"基金代码无需时效描述"。
    -- 你提取的数值信息应该完全保留精度信息，不做四舍五入或者省略处理。

    # step1:提取四元组的优质输出示例
    <回复答案>示例：
    今天半导体板块整体呈现震荡上行态势，截至2025年02月24日盘中休市，半导体指数涨幅+0.20%，最高触及3272.45点。从市场情绪来看，资金对半导体行业的关注度持续提升，板块内多只个股表现活跃。<card>15</card> 
    ### **一、市场表现分析** 
    **ETF产品热度提升**： 
    博时中证半导体产业ETF（159582）在上一交易日（2月21日）上涨3.91%，表明资金通过指数产品加速布局半导体行业。<citation>5</citation> 
    ### **二、行业动态与政策支持** 
    <card>14</card> 
    1.**政策驱动技术突破**： 
    上海市近期发布“算力生态超市”模速空间平台，推动人工智能与半导体技术协同发展。中芯京城二期项目落地北京亦庄，总投资超500亿元，预计将显著提升高端芯片产能。<citation>7</citation> 
    比如一家投资公司A手上有1000元的资金，打算投资半导体，预期收益1.5%，1年后收益如何可以再详细讨论一下。

    <提取的四元组>输出示例：
    [{"主体"："半导体指数", "指标"："涨幅", "数值"："+0.20%", "时间"：["截至2025年02月24日盘中休市"]}, {"主体"："半导体", "指标"："最高指数值", "数值"："3272.45", "时间"：["截至2025年02月24日盘中休市"]}, {"主体"："博时中证半导体产业ETF", "指标"："基金代码", "数值"：159582, "时间"：["基金代码无需时效描述"]}, {"主体"："博时中证半导体产业ETF", "指标"："涨幅", "数值"："3.91%", "时间"：["2025年2月21日"]}, {"主体"："中芯京城二期项目", "指标"："投资金额", "数值"："500亿元", "时间"：["未找到具体时间描述"]}, {"主体"：公司A, "指标"："投资金额", "数值"："1000元", "时间"：["举例假设信息无需明确时效-假定投资金额"]}, {"主体"："公司A", "指标"："预期收益", "数值"："1.5%", "时间"：["举例假设信息无需明确时效-假设预期收益"]},]

    # <回复答案>：你考试时需要针对一下回复答案进行提取：
    <回复答案替换>

    # 第二步判断提取的四元组信息是否和参考材料中表述的信息一致
    ## 四元组中错误类型可能包含如下问题：
    --**时间**错误：时效混淆错误，描述的主体/项目将当前时间、发布时间、和发布文章内部的时间混淆，把不同# 参考资料内容中不同的描述信息时间混淆；
    --**时间**缺失：时间信息缺失，提取的四元组中相关数值需要结合对应时效日期描述反应项目/数值的严谨性，# 参考资料中对指标有明确的时效日期，但是提取的四元组中中没有给出具体的时效日期；
    --**数值**幻觉：提取的四元组中，数值出现# 参考资料供给中没有出现的数值（备注：但与通识相关的数值如客服的一般规则，时间，金额或者投资建议相关的数值如建议投资比例等不属于此类。）
    --**数值**错误：数值由于四舍五入与#参考资料中数值不一致导致的错误；或者数值混淆错误，混淆错误是描述的主体/项目对应的数值错误的将# 参考资料中的其他主体/项目的数值映射到当前主体了。如将资金流入率的数值描述成资金净流入的数值；最大回撤的数值描述成动态回撤等。
    --**主体**错误：在描述事件时，对不同的事件和主体混淆，将某个对象的事件信息描述归因到错误的对象上。如 “沪深300主题当前涨幅为XXX”描述为“沪深300指数当前涨幅XXX” 

    ## 判断step1中<提取的金融四元组>的错误类型的一些特殊类型：（你在考试时需要注意以下一些特殊情形的判定）
    -- 基金/股票代码类**数值**不需要提供**时间**说明
    -- 与金融常识相关的**数值**，提取的**数值**不需要作严格比对，可以不提供**时间**说明，如【客服的一般规则、手续费、限额、持仓金额、建议时间等】
    -- <提取的金融四元组>中*时间*槽位包含“举例假设”前提时，提取的**数值**不需要作严格比对，数值不同可以判定为正确，计算得出的**数值**也不用严格来自# 参考资料，同时可以不提供**时间**说明
    -- 与投资建议相关的金融常识**指标**可以不用严格来自# 参考资料中的数值，也可以不提供**时间**说明，涉及的指标包括：如[XXX配置比例、XXX持仓比例、XXX分配比例等]
    -- 存在四舍五入、没有保留和# 参考资料中一样的精度**数值**描述，都属于**数值**错误（去除精度中末尾的0除外）。例如将基金规模为3019.20元省略四舍五入描述为约3千多元属于**数值**错误，可以表示为3019.2正确；

    ## step2:判断要求：提取的四元组是否和参考资料中一致
    1. 你需要对**每个**参考资料内容依次进行分析，找出其中所有与<提取的金融四元组>中**主体**、**指标**和**数值**相关的信息内容。单个主体/金融指标/数值数值可能关联多个信息片，你需要依次分析。
    2. 对你提取的四元组和给定的<提取的金融四元组>进行对比分析，按错误类型进行分类，并且说明错误原因，如果无错误则对应的类型和原因都输出'无'。
    3. @@注意每错误判断一个四元组都会被扣分，因此请仔细作答考试。

    # 输出格式要求
    1. 只需要输出第二步针对每个提取的元组判断结果即可，不要做其他额外解释，以json格式给出，格式如下：[{"主体"：xxx, "指标"：xxx, "数值"：xxx, "时间"：xxx, "错误类型"：xxx,"错误原因"： xxx},...]

    # 以下是你需要比对提取的四元组使用的参考资料：
    ## 参考材料：
    <参考材料>
    """
        substring = "# 参考资料"
        # count = prompt.count(substring)
        # ref = prompt
        # if count<2:
        #     # ref = prompt.split('# 参考资料')[1].split('# 当前问题')[0]
        #     ref = row['messages'][1]['content'][row['messages'][1]['content'].find('# 之前的聊天内容'):row['messages'][1]['content'].find('# 当前问题\n')]
        # else:
        #     ref = prompt.split('# 参考资料\n')[1].split('# 当前问题')[0]
        ref = prompt[prompt.find("# 之前的聊天内容") : prompt.find("# 当前问题\n")]
        answer = response.replace("\n\n\n", "\n").replace("\n\n", "\n")
        try:
            answer = answer.split("</think>")[1]
        except:
            try:
                answer = answer.split("<answer>")[1]
            except:
                answer = answer
        answer = (
            answer.replace("<think>", "")
            .replace("</think>", "")
            .replace("<answer>", "")
            .replace("</answer>", "")
            .replace("\n\n", "\n")
            .replace("\n需", "")
        )
        prompt_ref = (
            prompt_template.replace("<参考材料>", ref)
            .replace("<br/>", "\n")
            .replace("\n\n\n", "\n")
        )
        prompt_answer = (
            prompt_ref.replace("<回复答案替换>", answer)
            .replace("<br/>", "\n")
            .replace("\n\n\n", "\n")
        )
        item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_answer},
            ]
        }
        return item["messages"]

    # 提取时优先提绝对时间，没有绝对时间提相对时间，出现近X年没有具体时效日期特判
    def match_near_year(self, time):
        # 使用正则表达式匹配 "近*年"，其中 * 表示任意多个字符
        pattern = r"近.*年"
        match = re.search(pattern, time)
        return match is not None

    def parse_outputs(self, ans_judge):
        ans_judge_list_filter = []
        juliflag = 0
        ans_judge_list = ans_judge
        for i in ans_judge_list:
            if "时间" in i:
                time = i["时间"]
            else:
                try:
                    time = i["time"]
                except:
                    print("no time----")
                    print(i)
                    continue
            if self.match_near_year(str(time)):
                i.update({"错误类型": "时间缺失"})
                ans_judge_list_filter.append(i)
                continue

            if "错误类型" in i:
                err_type = i["错误类型"]
            elif "错误Type" in i:
                err_type = i["错误Type"]
            elif "error_type" in i:
                err_type = i["error_type"]
            else:
                try:
                    err_type = i["错误类型"]
                except:
                    print(i, "错误类型丢失")
                    continue
            # 持仓配置类跳过
            if "比例" in i["指标"]:
                continue
            # 无错误跳过
            if err_type == "无":
                continue
            # 举例假设类跳过
            if "举例" in time or "假设" in time:
                continue
            if "假设" in i["错误原因"] or "示例" in i["错误原因"]:
                continue
            ans_judge_list_filter.append(i)

        # print(ans_judge_list_filter)
        print("严谨性错误详情:", ans_judge_list_filter)
        return len(ans_judge_list_filter)
        # if len(ans_judge_list_filter)==0:
        #     return 1
        # else:
        #     return 0

    def get_api_reward_xh_yanjing(self, messages_info):
        # client = OpenAI(api_key="EMPTY",base_url="http://33.137.80.20:9122/v1")
        # if random.randint(0,6) > 3:
        #     completion = self.client_1.chat.completions.create(
        #     model="/mnt4/ckpt/dengkai.dk/reward_tuple/merged/faithful_reward_tuple_lora_zero2_0328_v1_ckpt_152",
        #     messages=messages_info,
        #     max_tokens=2000,
        #     )
        # else:
        completion = self.client_1.chat.completions.create(
            model="/mnt4/ckpt/dengkai.dk/reward_tuple/merged/faithful_reward_tuple_lora_zero2_0328_v1_ckpt_152",
            messages=messages_info,
            max_tokens=2000,
        )
        print(completion)
        message_content = completion.choices[0].message.content
        # 将JSON字符串转换为Python对象
        data = json.loads(message_content)
        return self.parse_outputs(data)

    def finance_yanjin_judge_ray(self, messages_info):
        #  "max_tokens": 3000,
        infer_data = {
            "__entry_point__": "openai.chat.completion",
            "model": "auto",
            "messages": messages_info,
            "stream": False,
            "temperature": 0,
            "repetition_penalty": 1,
            "top_p": 0.95,
            "top_k": 20,
            "max_token": 3000,
        }
        # 调用api
        # result = self.test_maya_stream(infer_data, api_name)
        result = self.test_ray(infer_data)
        if result != "":
            data = json.loads(result)
            return self.normalize_error_score_exp(
                self.parse_outputs(data), max_errors=8
            )
        else:
            return 0

    def normalize_error_score_exp(self, error_count, max_errors=10, decay_rate=0.3):
        """
        使用指数衰减的评分方式
        error_count: 实际错误数量
        max_errors: 最大错误数量阈值
        decay_rate: 衰减率，控制分数下降的速度
        """
        error_count = min(error_count, max_errors)

        # 使用指数函数计算分数
        score = math.exp(-decay_rate * error_count)

        return round(score, 2)

    def __call__(self, batch_data) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            batch_data (list[dict]): Generated outputs

        Returns:
            list[float]: Reward scores
        """
        def process_completion(data):
            messages = data["messages"]
            system_ = messages[0]["content"]
            prompt_ = messages[1]["content"]
            try:
                rewards = [
                    self.finance_yanjin_judge_ray(
                        self.build_extract_template(prompt_, data["solution"])
                    )
                ]
                print("严谨判断reward为", rewards)
                return rewards[0]
            except Exception as e:
                print("模型reward报错:", e)
                return 0.0

        with ThreadPoolExecutor(max_workers=min(len(batch_data), 10)) as executor:
            overall_reward = list(executor.map(process_completion, batch_data))

        return overall_reward


conclusion_prompt = """## 角色
你是一个金融领域的问答打分专家，根据问题和打分原则对回答进行打分

## 评价原则：
一、基础要求：
1. 结论完整且回应用户问题。结论能够解决用户问题，并且回答了query中的所有问题。
  * 如果用户问题无法回答或无法选出要求的产品，需要合理解释，并且给出替代方案，选股场景要给出具体产品
  * 不满足的例子：
    * 用户问题：分析下我赎回的博时标普500ETF联接(QDII)C，想找稳一点的机会 
      回答：只是分析了该基金，并没有针对用户要求，详细推品，且还在回答“对博时标普500基金，建议每月定投金额不超过权益类投资的30% ”
2. 结论明确。针对用户的问题，有明确的结论。
  * 选择基金的场景必须要有具体的产品/板块/基金经理
  * 判断股票的场景下，要有明确的总结，最好在结尾
  * 结论明确的例子：
    * 结合你的需求和当前市场环境，推荐你考虑选择XXXX基金
    * 如果该产品符合你的风险偏好和收益要求，你可以考虑购买一些
3. 结论一致。上下文结论一致，如果不一致要能自圆其说。

二、加分项：
1. 是否提供详实操作建议：如不同风险类型的用户分别适合哪些投资方案（无用户背景资料）或者是按照客户的背景应该投哪些产品（有用户背景资料）。或用户询问配置建议，明确给出来不同资产的配置金额或比例。
  * 不满足的例子：
    * 你可以根据自己的风险承受能力选择适合自己的产品。
    * 推荐了平台不可售的产品
  * 具体可执行的操作建议举例：
    * 如涉及到推荐产品，推荐的产品应该是平台可售卖的产品
    * 建议用户通过定投等方式参与
    * 给出具体各类产品的配置比例，比如稳健60%，进阶40%
    * 操作一定是要细化到具体可购买的产品
2. 是否提供风险提示：直接点明风险提示
3. 相关服务推荐：与用户query相关的其他增值类服务，例如选基的时候推荐“帮你投”等服务类型
4. 相关产品推荐：
  * 不满足的情况：如果问题是询问某产品，推荐了问题以外的产品，或者用户无明确产品需求，推荐了他问题相关的产品

## 得分规则：
- 基础要求每满足一条得1分，不满足得0分
- 基础要求有不满足的条款，加分项全为0分；基础要求全满足的情况下，加分项每满足一条得1分，不满足得0分

## 问题
<-query->

## 当前时间
<-time->

## 回答
<-answer->

## 目标
根据问题和回答，分析每一个评价原则和相应的得分，最后用json形式给出这个回答每项的得分和总的得分。

## 输出格式
【评价原则】：
1. 基础要求1：<分析每个原则>
2. 基础要求2：<分析每个原则>
3. 基础要求3：<分析每个原则>
4. 加分项1：<分析每个原则>
5. 加分项2：<分析每个原则>
6. 加分项3：<分析每个原则>
7. 加分项4：<分析每个原则>
【得分】：
{"基础要求1": x, "基础要求2": x, "基础要求3": x, "加分项1": x, "加分项2": x, "加分项3": x, "加分项4": x, "总分": x}
"""

argument_prompt = """
## 角色
你是一个金融领域的问答打分专家，根据问题和打分原则对回答进行打分

## 评价原则：
一、基础要求：
1. 论据多样性。对总论点给出2个以上的论据。
2. 论据时效性。论据应该与问题的时效性一致，如近期问题一定用近期论据，长期分析或者是过去的问题可以用时效性稍差论据。
3. 论据与结论一致性。使用的论据与整体结论的观点一致或能自圆其说。

二、加分项：
1. 论据高相关性。所有论据均与论点高度相关。
  - 直接关联：数据指标选取需反映论点核心要素（如论证 "货币政策宽松刺激消费" 时，应选用 M2 增速、社融增量等直接影响消费能力的指标，而非间接相关的工业增加值）
  - 范围匹配
    * 不满足的例子：
      空间维度：论证中国市场数据时，引用美国市场数据，但未说明关联
      市场层级：讨论主板市场流动性时，使用新三板交易数据属于市场分层错位
2. 论据有明确信息。所有论据都具有数据、指标、事件等明确信息。
  * 不满足的例子：泛泛而谈的“很多XX认为”“行业中多数企业”等表述
  * 满足的例子：使用“5家券商的分析师认为”“行业内70%的企业”
3. 论据层次性：各论据之间可以形成层层递进的逻辑链条，使得整个论证过程更加有说服力
  * 满足的例子：从市场到板块到个基，或者从个基到板块到整体市场。
    具体而言可以是整体市场向好，在此基础上该板块近期表现也尤为突出，与此同时该基金在短期与长期的收益率都排名靠前，所以推荐该基金
4. 论据逻辑性：论据与论点之间存在可验证的因果关系或逻辑推导链条，无断裂环节或牵强附会的间接联系。论据没有以偏概全、滑坡谬误、强行将相关性认为是因果关系，或者倒置因果关系等逻辑谬误。
  * 不满足的例子：
    * XX基金最大回撤排名靠前，所以该基金收益率很高
    * XX基金近一年净值创新高，体现出该基金风控能力优秀
    * 从XX公司近期营收上升较快推导出XX行业前景看好（以偏概全）

## 得分规则：
- 基础要求每满足一条得1分，不满足得0分
- 基础要求有不满足的条款，加分项全为0分；基础要求全满足的情况下，加分项每满足一条得1分，不满足得0分

## 问题
<-query->

## 当前时间
<-time->

## 回答
<-answer->

## 目标
根据问题和回答，分析每一个评价原则和相应的得分，最后用json形式给出这个回答每项的得分和总的得分。

## 输出格式
【评价原则】：
1. 基础要求1：<分析每个原则>
2. 基础要求2：<分析每个原则>
3. 基础要求3：<分析每个原则>
4. 加分项1：<分析每个原则>
5. 加分项2：<分析每个原则>
6. 加分项3：<分析每个原则>
7. 加分项4：<分析每个原则>
【得分】：
{"基础要求1": x, "基础要求2": x, "基础要求3": x, "加分项1": x, "加分项2": x, "加分项3": x, "加分项4": x, "总分": x}"""


express_prompt = """## 角色
你是一个金融领域的问答打分专家，根据问题和打分原则对回答进行打分

## 评价原则：
一、基础要求：
1. 文本符合自然语言要求。
  - 文本整体无语病，错别字等（包括重复信息如时间反复出现等）
  - 无内部信息透出（如根据网页xx"、"[webpage xx]"、"[service xx]"、"[card xx]"、"[data xx]"、"服务xx"、"卡片xx"、"图表xx"、"根据服务规范"、"根据金融监管要求"、"根据xx信息"(xx表示数字编号)等）
2. 图表位置恰当。图表出现位置不突兀有上下文承接。（如果没有图表默认此项得分）
3. 显示结果正常。显示呈现效果符合人的阅读习惯（如没有错误换行，子标题缩进及标识错误，字体大小差异等等）并且可以呈现所有信息
4. 重点信息一目了然。恰当的地方进行了加粗和高亮，但不能过多，全文总数控制在1-3个。

二、加分项：
1. 层级结构清晰。有恰当且排版正确的分点叙述结构。
2. 图表丰富。图表数量≥2
3. 表达降维易懂。能对专业名词进行降维表达，如alpha解释为市场超额收益。
4. 有markdown表格。

## 得分规则：
- 基础要求每满足一条得1分，不满足得0分
- 基础要求有不满足的条款，加分项全为0分；基础要求全满足的情况下，加分项每满足一条得1分，不满足得0分

## 说明
- 评价原则中提到的图表指的是引用的卡片，其格式为<card>使用描述:xxx;id:x</card>，可以根据其<使用描述>判断是否与上下文相关

## 问题
<-query->

## 当前时间
<-time->

## 回答
<-answer->

## 目标
根据问题和回答，分析每一个评价原则和相应的得分，最后用json形式给出这个回答每项的得分和总的得分。

## 输出格式
【评价原则】：
1. 基础要求1：<分析每个原则>
2. 基础要求2：<分析每个原则>
3. 基础要求3：<分析每个原则>
4. 基础要求4：<分析每个原则>
5. 加分项1：<分析每个原则>
6. 加分项2：<分析每个原则>
7. 加分项3：<分析每个原则>
8. 加分项4：<分析每个原则>
【得分】：
{"基础要求1": x, "基础要求2": x, "基础要求3": x, "基础要求4": x, "加分项1": x, "加分项2": x, "加分项3": x, "加分项4": x, "总分": x}"""

custom_prompt = """## 角色
你是一个金融领域的问答打分专家，根据问题和打分原则对回答进行打分

## 评价原则：
一、基础要求：
1. 结合用户持仓分析。能够结合用户的持仓信息进行论证。

二、加分项：
1. 结合用户基础信息。根据用户的年龄、性别、职业等给出有逻辑的分析。
2. 结合用户投资行为特征信息。根据用户的风险偏好、历史行为、投资经验等给出有逻辑的分析。

## 得分规则：
- 基础要求每满足一条得1分，不满足得0分
- 基础要求有不满足的条款，加分项全为0分；基础要求全满足的情况下，加分项每满足一条得1分，不满足得0分

## 问题
<-query->

## 当前时间
<-time->

## 回答
<-answer->

## 目标
根据问题和回答，分析每一个评价原则和相应的得分，最后用json形式给出这个回答每项的得分和总的得分。

## 输出格式
【评价原则】：
1. 基础要求1：<分析每个原则>
2. 加分项1：<分析每个原则>
3. 加分项2：<分析每个原则>
【得分】：
{"基础要求1": x, "加分项1": x, "加分项2": x, "总分": x}"""
# 结论实用、论据丰富、表达易懂、为我定制 基础分满分分别是 3、3、4、1
# 7、7、8、3

overall_prompt = """## 角色
你是一个金融领域的问答评价专家，根据问题，请从多个角度对回答进行综合评价

## 评价原则：
针对具体的问题，你需要评估回答内容的整体质量，以下是一些可供参考的评价原则，对于具体的问题和回答可以有不同的评价原则
1. 回答表述：回答内容表述清晰，无重复，格式混乱（包括markdown格式）等问题，正确使用特殊字段
2. 图文结合：回答中使用了表格或卡片信息，卡片表格上下文联系紧密
3. 回答明确：直接且精准回应用户的问题，不回避，不遗漏，可以结合用户持仓等信息给出具体的建议
4. 回答逻辑：回答逻辑自洽，论点和论据相互映证，前后观点一致
5. 回答结构：回答条理清晰，结构清晰，合理使用各级标题
6. 回答丰富：回答内容充实，论据扎实合理，有较好的时效性

## 打分规则：
- 你是一个严格的打分者，不要轻易给出高分，必须在各方面都完美时才可以给出很高的分数
- 根据评价原则，自行分析该回答每个原则对应的得分（推荐1-10分）和权重
- 你甚至可以自行增加一些你认为非常重要的原则，可以自行调整权重
- 列举每个原则的权重，并分析该回答的得分情况
- 根据每个原则的得分和权重，计算出一个得分S
- 最后用一个**整体体感折扣系数**alpha，乘以得分S，得到这个回答的综合评价得分
- 注意：你的目的是从**整体**上给出这个问题回答的**综合评价**，所以如果回答中出现了一些严重影响整体体感的内容，比如格式混乱，内容重复，完全答非所问等等，需要适当地调整**权重**和**整体体感折扣系数**，使得综合评价得分保持在一个低的水平（建议<=4分）

## 分数粒度
- 9-10分：表示完美符合评价原则
- 6-8分：表示回答符合大部分符合评价原则，但有一些遗漏
- 3-5分：表示回答部分符合评价原则，但主要部分未满足
- 1-2分：表示回答不符合评价原则

## 一些特殊字段
- 引用字段：<citation>xx</citation>，引用时需要使用该字段
- 卡片字段：<card>xx</card>
- 表格字段：<table>xx</table>
- 高亮字段：<highlight>xx</highlight>
- 结论字段：<conclusions>xx</conclusions>
- 回答字段：<answer>xx</answer>

## 问题
<-query->

## 当前时间
<-time->

## 回答
<-answer->

## 目标
根据问题，当前时间和回答，先分析这个评价这个回答需要哪些原则，注意不必局限于上述提到的评价原则，然后分析每一个评价原则和相应的得分与权重，最后从**整体**上给出这个问题回答的**综合评价**得分，得分不用四舍五入。

## 输出格式
【评价原则】：
1. <分析每个原则，权重，得分>
2. <分析每个原则，权重，得分>
【得分】：
这里给出总的得分：boxed{xx}"""
peek_dict = {
    "结论实用": {"min": 3, "max": 7, "prompt": conclusion_prompt},
    "论据丰富": {"min": 3, "max": 7, "prompt": argument_prompt},
    "表达易懂": {"min": 4, "max": 8, "prompt": express_prompt},
    "为我定制": {"min": 1, "max": 3, "prompt": custom_prompt},
    "整体得分": {"min": 1, "max": 10, "prompt": overall_prompt},
}


def parse_peek_outputs(output, _type):
    if _type == "整体得分":
        # output = row['output']
        pattern = r"boxed{([\d.]+)}"
        matches = re.findall(pattern, output)
        print(matches)
        return float(matches[-1])
    else:
        if "<answer>" in output:
            answer = output[
                output.find("<answer>") + len("<answer>") : output.find("</answer>")
            ].strip()
        else:
            answer = output[output.find("</think>") + len("</think>") :].strip()
        answer = answer.replace("\n", "").replace(" ", "")
        if "【得分】：" in answer:
            answer = answer[answer.find("【得分】：") + len("【得分】：") :].strip()
        if "```json" in answer:
            answer = (
                answer[answer.find("```json") + len("```json") : answer.find("}`") + 1]
                .strip()
                .strip("`")
            )
        print("answer:", answer)
        answer_json = json.loads(answer)
        total_score = answer_json["总分"]
        return total_score


def compute_peek_final(score, min_val=4, max_val=8):
    # 使用最大最小值归一化公式：(x - min) / (max - min)
    if score <= min_val:
        return 0.0
    elif score >= max_val:
        return 1.0
    else:
        return (score - min_val) / (max_val - min_val)


class Map_peek_based_openai(ORM):
    def __init__(
        self,
    ):
        self.ray_client = MayaClient.get_instance()
        self._type = "整体得分"

    def test_ray(self, infer_data):
        self.ray_request = ArksRequest()
        self.ray_request.scene_name = "tigan_reward"
        self.ray_request.chain_name = "v1"
        random_str = str(uuid.uuid4())
        # 去掉连字符
        session_id = str(uuid.uuid4()).replace("-", "")
        # print(session_id)
        self.ray_request.request_timeout = 60000
        self.ray_request.read_timeout = 60000
        self.ray_request.connect_timeout = 60000
        self.ray_request.load_balancing_policy = LoadBalanceStrategy.RANDOM
        # self.ray_request.session_id = "12424"
        self.ray_request.session_id = session_id
        self.ray_request.set_lbconfig({"arks.antvip.arrow.cross.city": "true"})
        self.ray_request.set_item_num(1)
        self.ray_request.router_priority = RouterPriority.CONFIG
        self.ray_request.set_serverlist(["33.134.18.115:10001"])
        # self.ray_request.item_number_per_packet = 1 # 发给每个后端请求的item个数
        # self.ray_request.max_packet_count = 1 # 这个request最多分为多少个发给后端的请求包
        # item_id = "123456789"
        # 生成 UUID
        random_str = str(uuid.uuid4())
        # 去掉连字符
        item_id = str(uuid.uuid4()).replace("-", "")
        # print(item_id)

        item = self.ray_request.add_item(item_id)
        tensor_key = "data"
        tensor_values = [json.dumps(infer_data, ensure_ascii=False)]
        print("输入数据", tensor_values)
        shapes = [1]
        item.add_tensor_feature(tensor_key, shapes, tensor_values)
        res = self.ray_client.call(self.ray_request)
        if res.success:
            print("request success")
            result_dict = res.items[0].attributes
            result_json_data = json.loads(result_dict["result"])
            content = result_json_data["choices"][0]["message"]["content"]
            if self._type == "整体得分":
                ovreall_score = parse_peek_outputs(content, self._type)
                overall = compute_peek_final(
                    ovreall_score,
                    peek_dict[self._type]["min"],
                    peek_dict[self._type]["max"],
                )
            else:
                ovreall_score = parse_peek_outputs(content, self._type)
                overall = compute_peek_final(
                    ovreall_score,
                    peek_dict[self._type]["min"],
                    peek_dict[self._type]["max"],
                )
            print("体感结果", overall)
            return overall
        else:
            print("体感 request fail")
            print(res.error_msg)
            return ""

    def build_extract_template(self, prompt, answer):
        prompt_template = peek_dict[self._type]["prompt"]
        answer = answer.replace("\n\n\n", "\n").replace("\n\n", "\n")
        try:
            if "</think>" in answer:
                answer = answer.split("</think>")[1]
            else:
                answer = answer
        except:
            answer = answer
        answer = (
            answer.replace("<think>", "")
            .replace("</think>", "")
            .replace("<answer>", "")
            .replace("</answer>", "")
            .replace("\n\n", "\n")
            .replace("\n需", "")
        )
        query = prompt[
            prompt.find("# 当前问题\n")
            + len("# 当前问题\n") : prompt.find("# 回答要求重点\n")
        ].strip()
        time = prompt[
            prompt.find("# 当前时间\n")
            + len("# 当前时间\n") : prompt.find("\n港交所交易状态:")
        ]
        prompt_answer = (
            prompt_template.replace("<-query->", query)
            .replace("<-answer->", answer)
            .replace("<-time->", time)
        )
        item = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt_answer},
            ]
        }
        return item["messages"]

    def finance_tigan_ray(self, messages_info):
        #  "max_tokens": 3000,
        infer_data = {
            "__entry_point__": "openai.chat.completion",
            "model": "auto",
            "messages": messages_info,
            "stream": False,
            "temperature": 0,
            "repetition_penalty": 1,
            "top_p": 0.95,
            "top_k": 20,
            "max_token": 3000,
        }
        # 调用api
        # result = self.test_maya_stream(infer_data, api_name)
        result = self.test_ray(infer_data)
        if result != "":
            return result
        else:
            return 0

    def __call__(self, batch_data) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            batch_data (list[dict]): Generated outputs

        Returns:
            list[float]: Reward scores
        """
        def process_completion(data):
            messages = data["messages"]
            system_ = messages[0]["content"]
            prompt_ = messages[1]["content"]
            try:
                rewards = [
                    self.finance_tigan_ray(self.build_extract_template(prompt_, data["solution"]))
                ]
                print("体感reward为", rewards)
                return rewards[0]
            except Exception as e:
                print("体感reward报错:", e)
                return 0.0

        with ThreadPoolExecutor(max_workers=min(len(batch_data), 10)) as executor:
            overall_reward = list(executor.map(process_completion, batch_data))

        return overall_reward


class Map_faithful_model_based_client(ORM):
    def __init__(
        self,
    ):
        # 33.184.122.245
        self.client_1 = OpenAI(
            api_key="EMPTY", base_url="http://33.184.122.245:8100/v1"
        )
        # self.client_2 = OpenAI(api_key="EMPTY",base_url="http://33.148.46.203:8100/v1")

    def build_extract_template(self, prompt, response):
        prompt_template = """# 任务(提取元组后判断类型和原因)
    你现在在进行金融信息提取和判断能力的考试，你需要经过严密的分析思考完成考试，考试分为两个步骤(最终结果按# 输出格式要求 输出)：
    第一步，提取出<回复答案>中所有和金融数值相关的四元组信息（主体，指标，数值，时间）；第二步，判断提取的四元组信息是否和# 参考材料中表述的信息一致。

    # 金融四元组的定义
    1. 主体：指标的主体，比如：XXX基金/股票/债券、XXX公司/行业、板块、人名、产品名等，保持提取的主体完整性和明确性
    2. 指标：金融指标的名称，详见**金融指标字典**
    3. 数值：指标的数值描述，可带正负号、单位、中文、千位逗号、比较词低于/高于/不少于等，比如：3.14、-0.5%、1万元、2,000手，高于1.3%等，保持提取的数值准确性
    4. 时间：指标的时效性，金融数值的时间属性，通常关联到一个特定的时间点、或者由起始和结束的时间段组成，比如：2025年03月04日，时间段2024年初至今

    # step1:提取金融四元组的具体要求
    1. 你需要对**每个**<回复答案>中的内容依次进行分析，对其中所有包含**数值**的信息提取出四元组信息。
    2. 提取的指标应该在**金融指标字典**内，或者是与金融指标字典相关的指标名称。
    3. 在满足回答要求2的基础上，注意每遗漏一个**数值**都会被扣分，请仔细作答。

    ## 金融指标字典
    1. 通用指标：[价格、涨跌额、涨跌幅、开盘价、收盘价、最高价、最低价、均价、当前价、XXX价、收益率、收益金额、年化收益率、累积收益率、持有收益率、超额收益率、XXX收益率、超额回撤、动态回撤、最大回撤、ROE、ROA、资金流入、资金流出、资金净流入、资金净流出、久期、胜率、赔率、XXX年利率、XXX额度、成交量、成交额、融资净买入、公允价值]
    2. 持仓指标：[配置比例、建议配置比例、持仓收益影响占比、持有金额、持有收益金额]
    3. 基金指标：[份额、净值、持仓前十占比、综合费率、托管费率、管理费率、基金代码、万份收益、单位净值、7日年化收益率、规模、跟踪误差、夏普比率、XXX年化收益率]
    4. 基金经理指标：[任期回报、年化回报、在管天数、管理规模、净值回报]
    5. 股票指标：[基本每股收益、每股净资产、主力资金净流入、游资资金净流入、散户资金净流入、超大单净流入、大单净流入、中小单净流入、换手率、流通股数、总股数、市值、市盈率、市净率、支撑位价格、目标价、每股净资产]
    6. 财报指标：[营业收入、净利润、经营活动产生的现金流量净额、研发费用、营业收入、资产负债率、毛利润、毛利率、流动负债、归母净利润、主营收入、扣非净利润、流动比率、速动比率、EBITDA、调整后EBITDA、业务收入、总收入、每股经营现金流、营销费用]
    7. 前缀：[平均、预期、建议、滚动]
    7.1 * 可拼接在1-6中列举的指标的前面，形成新的指标
    8. 后缀：[同比上升/上涨/增长、环比上升/上涨/增长、xx天移动平均、历史分位数、增速、中位数、排名]
    8.1 * 可拼接在1-6中列举的指标的后面，形成新的指标

    ## 提取金融四元组的特殊说明,你在考试时需要注意以下特殊情形
    -- 对与举例类型的数值，通常是假定/假设类信息，因此与之关联的数值提取的**时间**信息，需要提取为“举例假设信息无需明确时效”，涉及的关键词【当XXX、假设XXX、如果XXX、假定XXX、如XXX、比如XXX、例如XXX、若XXX、等】。
    -- 提取的**指标**在# 金融指标字典找不到或者存在模糊对应关系时，尽可能保留<回复答案>中原始指标名称。
    -- 你需要提取XXX股票/基金代码，股票/基金代码的**数值**通常由6位连续数字组成，通常有对应的股票/基金名，提取**指标**股票代码/基金代码时，提取的**时间**为"基金代码无需时效描述"。
    -- 你提取的数值信息应该完全保留精度信息，不做四舍五入或者省略处理。

    # step1:提取四元组的优质输出示例
    <回复答案>示例：
    今天半导体板块整体呈现震荡上行态势，截至2025年02月24日盘中休市，半导体指数涨幅+0.20%，最高触及3272.45点。从市场情绪来看，资金对半导体行业的关注度持续提升，板块内多只个股表现活跃。<card>15</card> 
    ### **一、市场表现分析** 
    **ETF产品热度提升**： 
    博时中证半导体产业ETF（159582）在上一交易日（2月21日）上涨3.91%，表明资金通过指数产品加速布局半导体行业。<citation>5</citation> 
    ### **二、行业动态与政策支持** 
    <card>14</card> 
    1.**政策驱动技术突破**： 
    上海市近期发布“算力生态超市”模速空间平台，推动人工智能与半导体技术协同发展。中芯京城二期项目落地北京亦庄，总投资超500亿元，预计将显著提升高端芯片产能。<citation>7</citation> 
    比如一家投资公司A手上有1000元的资金，打算投资半导体，预期收益1.5%，1年后收益如何可以再详细讨论一下。

    <提取的四元组>输出示例：
    [{"主体"："半导体指数", "指标"："涨幅", "数值"："+0.20%", "时间"：["截至2025年02月24日盘中休市"]}, {"主体"："半导体", "指标"："最高指数值", "数值"："3272.45", "时间"：["截至2025年02月24日盘中休市"]}, {"主体"："博时中证半导体产业ETF", "指标"："基金代码", "数值"：159582, "时间"：["基金代码无需时效描述"]}, {"主体"："博时中证半导体产业ETF", "指标"："涨幅", "数值"："3.91%", "时间"：["2025年2月21日"]}, {"主体"："中芯京城二期项目", "指标"："投资金额", "数值"："500亿元", "时间"：["未找到具体时间描述"]}, {"主体"：公司A, "指标"："投资金额", "数值"："1000元", "时间"：["举例假设信息无需明确时效-假定投资金额"]}, {"主体"："公司A", "指标"："预期收益", "数值"："1.5%", "时间"：["举例假设信息无需明确时效-假设预期收益"]},]

    # <回复答案>：你考试时需要针对一下回复答案进行提取：
    <回复答案替换>

    # 第二步判断提取的四元组信息是否和参考材料中表述的信息一致
    ## 四元组中错误类型可能包含如下问题：
    --**时间**错误：时效混淆错误，描述的主体/项目将当前时间、发布时间、和发布文章内部的时间混淆，把不同# 参考资料内容中不同的描述信息时间混淆；
    --**时间**缺失：时间信息缺失，提取的四元组中相关数值需要结合对应时效日期描述反应项目/数值的严谨性，# 参考资料中对指标有明确的时效日期，但是提取的四元组中中没有给出具体的时效日期；
    --**数值**幻觉：提取的四元组中，数值出现# 参考资料供给中没有出现的数值（备注：但与通识相关的数值如客服的一般规则，时间，金额或者投资建议相关的数值如建议投资比例等不属于此类。）
    --**数值**错误：数值由于四舍五入与#参考资料中数值不一致导致的错误；或者数值混淆错误，混淆错误是描述的主体/项目对应的数值错误的将# 参考资料中的其他主体/项目的数值映射到当前主体了。如将资金流入率的数值描述成资金净流入的数值；最大回撤的数值描述成动态回撤等。
    --**主体**错误：在描述事件时，对不同的事件和主体混淆，将某个对象的事件信息描述归因到错误的对象上。如 “沪深300主题当前涨幅为XXX”描述为“沪深300指数当前涨幅XXX” 

    ## 判断step1中<提取的金融四元组>的错误类型的一些特殊类型：（你在考试时需要注意以下一些特殊情形的判定）
    -- 基金/股票代码类**数值**不需要提供**时间**说明
    -- 与金融常识相关的**数值**，提取的**数值**不需要作严格比对，可以不提供**时间**说明，如【客服的一般规则、手续费、限额、持仓金额、建议时间等】
    -- <提取的金融四元组>中*时间*槽位包含“举例假设”前提时，提取的**数值**不需要作严格比对，数值不同可以判定为正确，计算得出的**数值**也不用严格来自# 参考资料，同时可以不提供**时间**说明
    -- 与投资建议相关的金融常识**指标**可以不用严格来自# 参考资料中的数值，也可以不提供**时间**说明，涉及的指标包括：如[XXX配置比例、XXX持仓比例、XXX分配比例等]
    -- 存在四舍五入、没有保留和# 参考资料中一样的精度**数值**描述，都属于**数值**错误（去除精度中末尾的0除外）。例如将基金规模为3019.20元省略四舍五入描述为约3千多元属于**数值**错误，可以表示为3019.2正确；

    ## step2:判断要求：提取的四元组是否和参考资料中一致
    1. 你需要对**每个**参考资料内容依次进行分析，找出其中所有与<提取的金融四元组>中**主体**、**指标**和**数值**相关的信息内容。单个主体/金融指标/数值数值可能关联多个信息片，你需要依次分析。
    2. 对你提取的四元组和给定的<提取的金融四元组>进行对比分析，按错误类型进行分类，并且说明错误原因，如果无错误则对应的类型和原因都输出'无'。
    3. @@注意每错误判断一个四元组都会被扣分，因此请仔细作答考试。

    # 输出格式要求
    1. 只需要输出第二步针对每个提取的元组判断结果即可，不要做其他额外解释，以json格式给出，格式如下：[{"主体"：xxx, "指标"：xxx, "数值"：xxx, "时间"：xxx, "错误类型"：xxx,"错误原因"： xxx},...]

    # 以下是你需要比对提取的四元组使用的参考资料：
    ## 参考材料：
    <参考材料>
    """
        substring = "# 参考资料"
        # count = prompt.count(substring)
        # ref = prompt
        # if count<2:
        #     # ref = prompt.split('# 参考资料')[1].split('# 当前问题')[0]
        #     ref = row['messages'][1]['content'][row['messages'][1]['content'].find('# 之前的聊天内容'):row['messages'][1]['content'].find('# 当前问题\n')]
        # else:
        #     ref = prompt.split('# 参考资料\n')[1].split('# 当前问题')[0]
        ref = prompt[prompt.find("# 之前的聊天内容") : prompt.find("# 当前问题\n")]
        answer = response.replace("\n\n\n", "\n").replace("\n\n", "\n")
        try:
            answer = answer.split("</think>")[1]
        except:
            try:
                answer = answer.split("<answer>")[1]
            except:
                answer = answer
        answer = (
            answer.replace("<think>", "")
            .replace("</think>", "")
            .replace("<answer>", "")
            .replace("</answer>", "")
            .replace("\n\n", "\n")
            .replace("\n需", "")
        )
        prompt_ref = (
            prompt_template.replace("<参考材料>", ref)
            .replace("<br/>", "\n")
            .replace("\n\n\n", "\n")
        )
        prompt_answer = (
            prompt_ref.replace("<回复答案替换>", answer)
            .replace("<br/>", "\n")
            .replace("\n\n\n", "\n")
        )
        item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_answer},
            ]
        }
        return item["messages"]

    # 提取时优先提绝对时间，没有绝对时间提相对时间，出现近X年没有具体时效日期特判
    def match_near_year(self, time):
        # 使用正则表达式匹配 "近*年"，其中 * 表示任意多个字符
        pattern = r"近.*年"
        match = re.search(pattern, time)
        return match is not None

    def parse_outputs(self, ans_judge):
        ans_judge_list_filter = []
        juliflag = 0
        ans_judge_list = ans_judge
        for i in ans_judge_list:
            if "时间" in i:
                time = i["时间"]
            else:
                try:
                    time = i["time"]
                except:
                    print("no time----")
                    print(i)
                    continue
            if self.match_near_year(str(time)):
                i.update({"错误类型": "时间缺失"})
                ans_judge_list_filter.append(i)
                continue

            if "错误类型" in i:
                err_type = i["错误类型"]
            elif "错误Type" in i:
                err_type = i["错误Type"]
            elif "error_type" in i:
                err_type = i["error_type"]
            else:
                try:
                    err_type = i["错误类型"]
                except:
                    print(i, "错误类型丢失")
                    continue
            # 持仓配置类跳过
            if "比例" in i["指标"]:
                continue
            # 无错误跳过
            if err_type == "无":
                continue
            # 举例假设类跳过
            if "举例" in time or "假设" in time:
                continue
            if "假设" in i["错误原因"] or "示例" in i["错误原因"]:
                continue
            ans_judge_list_filter.append(i)

        # print(ans_judge_list_filter)
        print("严谨性错误详情:", ans_judge_list_filter)
        return len(ans_judge_list_filter)
        # if len(ans_judge_list_filter)==0:
        #     return 1
        # else:
        #     return 0

    def get_api_reward_xh_yanjing(self, messages_info):
        # if random.randint(0,6) > 3:
        completion = self.client_1.chat.completions.create(
            model="faithful_reward_tuple_lora_zero2_0328_v1_ckpt_152",
            messages=messages_info,
            temperature=0,
            max_tokens=3000,
        )
        # else:
        #     completion = self.client_2.chat.completions.create(
        #     model="faithful_reward_tuple_lora_zero2_0328_v1_ckpt_152",
        #     messages=messages_info,
        #     temperature=0,
        #     max_tokens=3000
        #     )
        print(completion)
        message_content = completion.choices[0].message.content
        # 将JSON字符串转换为Python对象
        data = json.loads(message_content)
        return self.normalize_error_score_exp(self.parse_outputs(data), max_errors=8)

    # def finance_yanjin_judge_ray(self, messages_info):
    #     #  "max_tokens": 3000,

    #     # 调用api
    #     # result = self.test_maya_stream(infer_data, api_name)
    #     result = self.test_ray(infer_data)
    #     if result != "":
    #         data = json.loads(result)
    #         return self.normalize_error_score_exp(
    #             self.parse_outputs(data), max_errors=8
    #         )
    #     else:
    #         return 0

    def normalize_error_score_exp(self, error_count, max_errors=10, decay_rate=0.3):
        """
        使用指数衰减的评分方式
        error_count: 实际错误数量
        max_errors: 最大错误数量阈值
        decay_rate: 衰减率，控制分数下降的速度
        """
        error_count = min(error_count, max_errors)

        # 使用指数函数计算分数
        score = math.exp(-decay_rate * error_count)

        return round(score, 2)

    def __call__(self, batch_data) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            batch_data (list[dict]): Generated outputs

        Returns:
            list[float]: Reward scores
        """
        def process_completion(data):
            messages = data["messages"]
            system_ = messages[0]["content"]
            prompt_ = messages[1]["content"]
            try:
                rewards = [
                    self.get_api_reward_xh_yanjing(
                        self.build_extract_template(prompt_, data["solution"])
                    )
                ]
                print("严谨reward为", rewards)
                return rewards[0]
            except Exception as e:
                print("严谨reward报错:", e)
                return 0.0

        with ThreadPoolExecutor(max_workers=min(len(batch_data), 10)) as executor:
            overall_reward = list(executor.map(process_completion, batch_data))

        return overall_reward


if __name__ == "__main__":
    orms = {}
    with open("./ant_rl_train_data_0509.json", "r") as f:
        test_data = json.load(f)

    # 不可用
    # orm = Map_faithful_model_based_openai()
    # print(orm(test_data[:2]))

    # 可用
    orm = Map_faithful_model_based_client()
    print(orm(test_data[:2]))

    orm = Map_peek_based_openai()
    print(orm(test_data[:2]))
