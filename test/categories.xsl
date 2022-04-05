<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
	<title></title>
	<style type='text/css'>
          body {
          font-family: "Candara", sans-serif;
          font-size: 9pt;
          line-height: 10.5pt;
          }
          div.pic h3 {
          font-size: 11pt;
          margin: 0.5em 1em 0.2em 1em;
          }
          div.pic p {
          font-size: 11pt;
          margin: 0.2em 1em 0.1em 1em;
          }
          div.pic {
          display: block;
          float: left;
          background-color: white;
          border: 1px solid #ccc;
          padding: 2px;
          text-align: center;
          margin: 2px 10px 10px 2px;
          -moz-box-shadow: 7px 5px 5px rgb(80,80,80);    /* Firefox 3.5 */
          -webkit-box-shadow: 7px 5px 5px rgb(80,80,80); /* Chrome, Safari */
          box-shadow: 7px 5px 5px rgb(80,80,80);         /* New browsers */
          }
          a { text-decoration: none; color: rgb(80,0,0); }
          a:hover { text-decoration: underline; color: rgb(255,80,80); }
          div.dirlinks h2 {  margin-bottom: 4pt; margin-left: -24pt; color: rgb(80,0,0);  }
          div.dirlinks {  margin: 0 24pt; }
          div.dirlinks a {
          font-size: 11pt; font-weight: bold;
          padding: 0 0.5em;
          }

          .categorylink{
          font-size: 20pt;
          font-weight: bold;
          height: 50px;
          display: inline-block;
          margin-top: 30px;
          }
          .categorydiv{
          float: left;
          }

	</style>
	<script language="JavaScript" implements-prefix="jscript">
	  var xslStylesheet;
	  var xslloaded = false;
	  var xsltProcessor = new XSLTProcessor();
	  var myDOM;

	  var xmlDoc;
	  var xslRef;
	  
	  function setcat(new_cat){
	  if (!xslloaded){
	  p = new XMLHttpRequest();
	  p.open("GET", "/hbakhshi/SMP-19-005/category.xsl", false);
	  p.send(null);
	  
	  xslRef = p.responseXML;
	  xsltProcessor.importStylesheet(xslRef);
	  xslloaded = true;

	  // load the xml file, example1.xml
	  myXMLHTTPRequest = new XMLHttpRequest();
	  myXMLHTTPRequest.open("GET", "data.xml", false);
	  myXMLHTTPRequest.send(null);
	  xmlDoc = myXMLHTTPRequest.responseXML;
	  }
	  //xsltProcessor.clearParameters();
	  xsltProcessor.setParameter(null, "selectedcategory" , new_cat );
	  document.title = xsltProcessor.getParameter(null, "selectedcategory");
	  var fragment = xsltProcessor.transformToFragment(xmlDoc, document);

	  
	  document.getElementById("cate_display").innerHTML = "";

	  myDOM = fragment;
	  document.getElementById("cate_display").appendChild(fragment);
	  }
	</script>
      </head>
      <body>
	<h1>Select a category to show plots:</h1>
	<xsl:for-each select="categories/category">
	  <a href="#" onclick="setcat('&quot;div' , name , '&quot;')" style="width=100px;display:block;">
	    <xsl:attribute name="onclick">
	      <xsl:value-of select="concat('setcat(&quot;' , name , '&quot;)' )"/>
	    </xsl:attribute>
	    <xsl:value-of select="name"/>
	  </a>
	</xsl:for-each>
	<div id='cate_display'>
	</div>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
