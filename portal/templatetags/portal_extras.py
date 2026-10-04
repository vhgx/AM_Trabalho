import markdown
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='markdownify')
def markdownify(value):
    """
    Converte uma string Markdown em HTML seguro.
    Inclui suporte para tabelas.
    """
    if not value:
        return ""
    
    # Ativa extensões de tabela e outras úteis
    html = markdown.markdown(value, extensions=['tables', 'fenced_code', 'nl2br'])
    return mark_safe(html)
