"""只读检查待提交源码和说明；仅使用标准库，不运行实验或加载模型。"""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def git_paths(*args):
    output = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z', *args])
    return [p for p in output.decode('utf-8').split('\0') if p]


def main():
    # 包括白名单内尚未暂存的新文件；不读取被忽略的报告、密钥或历史结果。
    files = sorted(set(git_paths()) | set(git_paths('--others', '--exclude-standard')))
    ignored_tracked = set(git_paths('--cached', '--ignored', '--exclude-standard'))
    errors = [f'{name}: 已被跟踪，但不在提交白名单' for name in sorted(ignored_tracked)]
    repo_paths = {ROOT / name for name in files if name not in ignored_tracked}
    link_targets = repo_paths | {parent for path in repo_paths for parent in path.parents if parent.is_relative_to(ROOT)}
    counts = dict(python=0, notebooks=0, json=0, markdown=0)
    allowed_suffixes = {'.py', '.ipynb', '.json', '.md', '.sh', '.ps1', '.txt'}
    forbidden_parts = {'.venv', 'models', 'vendor', 'cache', 'outputs', 'output', 'logs', '__pycache__',
                       'runs_distill', 'report_qa'}
    secret = re.compile(r'(?<![\w-])(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,})')
    link = re.compile(r'(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)')
    for name in files:
        if name in ignored_tracked:
            continue
        path = ROOT / name
        if not path.is_file():
            errors.append(f'{name}: 被跟踪文件不存在')
            continue
        if (path.suffix not in allowed_suffixes and path.name not in {'.gitignore', '.gitattributes'}) or forbidden_parts.intersection(path.relative_to(ROOT).parts):
            errors.append(f'{name}: 含不应提交的文件类型或目录')
            continue
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        if secret.search(text):
            errors.append(f'{name}: 疑似真实凭据，请人工检查；检查器不回显内容')
        if path.suffix == '.py':
            ast.parse(text, filename=name)
            counts['python'] += 1
        elif path.suffix in {'.json', '.ipynb'}:
            data = json.loads(text)
            if path.suffix == '.json':
                counts['json'] += 1
            else:
                counts['notebooks'] += 1
                for i, cell in enumerate(data['cells'], 1):
                    if cell.get('attachments'):
                        errors.append(f'{name}: 第{i}格含嵌入附件')
                    if cell['cell_type'] != 'code':
                        continue
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        errors.append(f'{name}: 第{i}格含运行输出或执行计数')
                    ast.parse(''.join(cell['source']), filename=f'{name}:cell{i}')
        elif path.suffix == '.md':
            counts['markdown'] += 1
            for target in link.findall(text):
                target = target.strip().strip('<>')
                if not target or target.startswith('#') or re.match(r'[A-Za-z][\w+.-]*:', target):
                    continue
                local = unquote(target.split('#', 1)[0])
                if local and (path.parent / local).resolve() not in link_targets:
                    errors.append(f'{name}: 本地链接目标未纳入仓库: {local}')
        elif path.suffix == '.sh' and b'\r\n' in raw:
            errors.append(f'{name}: Bash脚本含CRLF，应按.gitattributes保存为LF')
        if path.name.startswith('requirements'):
            for line in text.splitlines():
                if line.startswith('-r ') and (path.parent / line[3:].strip()).resolve() not in repo_paths:
                    errors.append(f'{name}: 引用的依赖清单未纳入仓库')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'PASS: {len(files)}个待管理文件；{counts["python"]}个Python程序，{counts["notebooks"]}个无输出Notebook，'
          f'{counts["json"]}个JSON配置，{counts["markdown"]}份Markdown说明。')
    print('语法、相对链接、依赖清单、Bash换行与提交边界检查通过；没有运行实验、下载资源或写入结果。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
