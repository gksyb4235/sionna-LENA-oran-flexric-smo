db = db.getSiblingDB("knowledge");
if (!db.getUser("jaechan_registry")) {
  db.createUser({
    user: "jaechan_registry",
    pwd: "f91AJMXNry6xxmm_99zt-dMCBXH_QbMAUoFeDxcbjZc",
    roles: [{ role: "readWrite", db: "knowledge" }]
  });
}
