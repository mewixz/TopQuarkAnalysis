<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns="http://www.w3.org/1999/xhtml" xmlns:html="http://www.w3.org/1999/xhtml" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" indent="yes" />
  <xsl:param name="selectedcategory" />
  <xsl:param name="myOrder" select="title" />
  <xsl:template match="/">
    <xsl:for-each select="categories/category[name=$selectedcategory]">
      <p><a href="#" class="categorylink">
	Category: <xsl:value-of select="name"/>
      </a></p>
      <div class="categorydiv" style="display:inline;">
	<xsl:attribute name="id">
	  <xsl:value-of select="concat('div' , name)"/>
	</xsl:attribute>
	<xsl:for-each select="variables/variable">
	  <div class='pic'>
	    <h3><a>
	      <xsl:attribute name="href">
		<xsl:value-of select="image"/>
	      </xsl:attribute>
	      <xsl:value-of select="image"/>
	    </a></h3>
	    <a>
	      <xsl:attribute name="href">
		<xsl:value-of select="image"/>
	      </xsl:attribute>
	      <img style="border: none; width: 400px;">
		<xsl:attribute name="src">
		  <xsl:value-of select="image"/>
		</xsl:attribute>
	      </img>
	    </a>
	  </div>
	</xsl:for-each>
      </div>
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
