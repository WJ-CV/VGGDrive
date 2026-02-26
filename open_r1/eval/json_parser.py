import re
import json

def fix_missing_commas(json_str):
    lines = json_str.split('\n')
    for i in range(len(lines) - 1):
        current_line = lines[i]
        if current_line.rstrip().endswith(','):
            continue
        next_line = lines[i + 1]
        # 计算前导\t数量
        current_tabs = len(current_line) - len(current_line.lstrip('\t'))
        next_tabs = len(next_line) - len(next_line.lstrip('\t'))
        if current_tabs != next_tabs:
            continue
        # 检查下一行是否以}或]开头
        next_strip = next_line.lstrip()
        if next_strip.startswith(('}', ']')):
            continue
        # 是否为键值对
        if ':' in current_line:
            lines[i] = current_line.rstrip() + ','
    return '\n'.join(lines)


class JsonParserFilter(object):
    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, response, **kwargs):
        if isinstance(response, list):
            response = response[0]

        content = response.strip()
        content = content.replace('```json', '').replace('```', '')
        content = fix_missing_commas(content) # 补全漏掉的逗号
        content = content.replace('\\"', '"').replace("\\'", "'").replace('\n', ' ')
        content = re.sub(r"(\w+)\s*:", r'"\1":', content)  # 处理无引号key
        content = re.sub(r"'", '"', content)  # 统一引号格式
        content = re.sub(r',(\s*[]}])', r'\1', content)  # 移除尾部逗号

        try:
            parsed = json.loads(content, strict=False)
        except Exception as e1:
            try:
                content = re.sub(r'\[([\d\s.,]+)\]',
                            lambda m: '[' + ','.join([f'[{x}]' if not x.strip().startswith('[') else x for x in m.group(1).split(',')]) + ']',
                            content)
                parsed = json.loads(content, strict=False)
                if not isinstance(parsed, dict):
                    parsed = {"lateral_control": "", "longitudinal_control": "", "trajectory": []}
            except Exception as e2:
                print(f'Json parser error: {e1}; {e2}')
                print([content])
                parsed = {"lateral_control": "", "longitudinal_control": "", "trajectory": []}

        trajectory_list = []
        for point in parsed.get("trajectory", []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            try:
                x = float(point[0])
                y = float(point[1])
                trajectory_list.append([x, y])
            except:
                continue

        return {
            'filtered_response': {
                "lateral_control": str(parsed.get("lateral_control", parsed.get("lateral control", ""))),
                "longitudinal_control": str(parsed.get("longitudinal_control", parsed.get("longitudinal control", ""))),
                "trajectory": trajectory_list[:6]  # 保持最大6个点的限制
            },
            'is_filtered': True if len(trajectory_list) > 0 else False
        }
