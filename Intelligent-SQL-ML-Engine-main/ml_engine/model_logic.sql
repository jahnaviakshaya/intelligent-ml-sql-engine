CASE
  WHEN "Torque [Nm]" <= 65.0000 THEN
  CASE
    WHEN "Torque [Nm]" <= 13.4500 THEN
    CASE
      WHEN "Torque [Nm]" <= 12.5500 THEN
        1
      ELSE
      CASE
        WHEN "Rotational speed [rpm]" <= 2601.5000 THEN
        CASE
          WHEN "Air temperature [K]" <= 297.5500 THEN
            0
          ELSE
            1
        END
        ELSE
          0
      END
    END
    ELSE
    CASE
      WHEN "Rotational speed [rpm]" <= 1380.5000 THEN
      CASE
        WHEN "Air temperature [K]" <= 301.5500 THEN
        CASE
          WHEN "Tool wear [min]" <= 188.5000 THEN
            0
          ELSE
            0
        END
        ELSE
        CASE
          WHEN "Process temperature [K]" <= 310.5500 THEN
            1
          ELSE
            0
        END
      END
      ELSE
      CASE
        WHEN "Torque [Nm]" <= 61.1000 THEN
        CASE
          WHEN "Tool wear [min]" <= 204.5000 THEN
            0
          ELSE
            0
        END
        ELSE
        CASE
          WHEN "Process temperature [K]" <= 310.4500 THEN
            1
          ELSE
            0
        END
      END
    END
  END
  ELSE
  CASE
    WHEN "Rotational speed [rpm]" <= 1217.0000 THEN
    CASE
      WHEN "Rotational speed [rpm]" <= 1201.0000 THEN
      CASE
        WHEN "Air temperature [K]" <= 301.1000 THEN
          1
        ELSE
          0
      END
      ELSE
        0
    END
    ELSE
    CASE
      WHEN "Type_M" <= 0.5000 THEN
      CASE
        WHEN "Tool wear [min]" <= 41.0000 THEN
        CASE
          WHEN "Torque [Nm]" <= 66.4000 THEN
            0
          ELSE
            1
        END
        ELSE
          1
      END
      ELSE
      CASE
        WHEN "Torque [Nm]" <= 67.8000 THEN
        CASE
          WHEN "Rotational speed [rpm]" <= 1299.5000 THEN
            0
          ELSE
            1
        END
        ELSE
          1
      END
    END
  END
END