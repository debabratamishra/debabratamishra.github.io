---
layout: null
permalink: /rss.xsl
sitemap: false
---
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:atom="http://www.w3.org/2005/Atom">
  <xsl:output method="html" indent="no"/>
  <xsl:template match="/">
    <html>
      <head>
        <title><xsl:value-of select="/rss/channel/title"/> RSS Feed</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #333; line-height: 1.6; }
          h1 { border-bottom: 1px solid #eee; padding-bottom: .5rem; }
          .item { margin-bottom: 2rem; }
          .item h2 { margin: 0 0 .25rem; }
          .item h2 a { color: #0066cc; text-decoration: none; }
          .item h2 a:hover { text-decoration: underline; }
          .date { color: #888; font-size: .85rem; }
          .desc { color: #555; }
        </style>
      </head>
      <body>
        <h1><xsl:value-of select="/rss/channel/title"/></h1>
        <p><xsl:value-of select="/rss/channel/description"/></p>
        <xsl:for-each select="/rss/channel/item">
          <div class="item">
            <h2><a href="{link}"><xsl:value-of select="title"/></a></h2>
            <span class="date"><xsl:value-of select="pubDate"/></span>
            <p class="desc"><xsl:value-of select="description"/></p>
          </div>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
