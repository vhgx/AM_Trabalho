from django.contrib import admin
from .models import Requisito


class RequisitoAdmin(admin.ModelAdmin):
    list_display = ('titulo', 'processado', 'criado_em', 'atualizado_em')
    list_filter = ('processado',)
    search_fields = ('titulo', 'descricao')
    readonly_fields = ('criado_em', 'atualizado_em')


admin.site.register(Requisito, RequisitoAdmin)
